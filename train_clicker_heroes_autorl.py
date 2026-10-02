"""
Script Principal para la Fase 3: Integración RL + AutoRL en Clicker Heroes con Parada de Emergencia.

Uso:
  - Verificar detección de ventana primero:
      python test_window_detection.py

  - Optimizar hiperparámetros con AutoRL (Optuna):
      python train_clicker_heroes_autorl.py --mode tune --algo PPO --trials 10 --timesteps 20000

  - Entrenar agente en modo seguro (Mock):
      python train_clicker_heroes_autorl.py --mode train --algo PPO --timesteps 50000

  - Entrenar agente en modo EN VIVO (con juego abierto y parada por tecla ESC):
      python train_clicker_heroes_autorl.py --mode train --algo PPO --timesteps 50000 --live
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import argparse
from typing import Optional, Dict, Any, Tuple, List, Callable
import optuna



from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy

from src.environments.clicker_heroes_env import ClickerHeroesEnv
from src.models.cnn_feature_extractor import CustomCNNFeatureExtractor
from src.utils.safety import EmergencyStopCallback, EmergencyStopException, global_emergency_listener
from stable_baselines3.common.monitor import Monitor
from src.utils.window_finder import WindowNotFoundError


def make_clicker_env(img_size: int = 84, mock_mode: bool = True):
    """Retorna una función creadora de ClickerHeroesEnv envuelta en Monitor para DummyVecEnv."""
    def _thunk():
        env = ClickerHeroesEnv(
            target_size=(img_size, img_size),
            max_steps=200,
            action_delay=0.03,
            mock_mode=mock_mode,
        )
        return Monitor(env)
    return _thunk



def build_stacked_env(img_size: int = 84, frame_stack: int = 4, mock_mode: bool = True):
    """Construye un entorno vectorizado con apilamiento de fotogramas (Frame Stacking)."""
    vec_env = DummyVecEnv([make_clicker_env(img_size=img_size, mock_mode=mock_mode)])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="first")
    return vec_env


import numpy as np
from optuna.pruners import MedianPruner
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


import datetime
import json

class TrialProgressCallback(BaseCallback):
    """
    Barra de progreso visual interactiva con seguimiento de Zonas/Niveles en tiempo real,
    HISTORIAL DE EVENTOS en vivo y PODA (Pruning) automática.
    """
    def __init__(
        self,
        total_timesteps: int,
        trial_num: int,
        total_trials: int,
        trial: Optional[optuna.Trial] = None,
        prune_interval: int = 500,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.trial_num = trial_num
        self.total_trials = total_trials
        self.trial = trial
        self.prune_interval = prune_interval
        self.pbar = None
        self.last_step = 0
        self.last_prune_check = 0
        self.last_heartbeat_step = 0
        self.log_file = os.path.join("logs", "clicker_heroes_history.log")
        self.events_file = os.path.join("logs", "clicker_heroes_events.jsonl")
        os.makedirs("logs", exist_ok=True)

    def _on_training_start(self) -> None:
        desc = f"Trial [{self.trial_num + 1}/{self.total_trials}]"
        self.pbar = tqdm(total=self.total_timesteps, desc=desc, unit="step", dynamic_ncols=True, leave=True)
        self.last_step = 0
        self.last_prune_check = 0
        self.last_heartbeat_step = 0
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_msg = f"\n[{now_str}] 🚀 SESIÓN INICIADA: {self.total_timesteps} pasos programados (Trial #{self.trial_num})"
        tqdm.write(start_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(start_msg + "\n")

    def _on_step(self) -> bool:
        delta = self.num_timesteps - self.last_step
        if delta > 0:
            self.last_step = self.num_timesteps
            try:
                venv = self.training_env
                while hasattr(venv, "venv"):
                    venv = venv.venv
                env_inst = venv.envs[0].unwrapped
                current_zone = getattr(env_inst, "current_zone", 1)
                max_zone = getattr(env_inst, "max_zone_reached", 1)
                last_rew = getattr(env_inst, "last_step_reward", 0.0)
                ep_rew = getattr(env_inst, "episode_total_reward", 0.0)
                last_act = getattr(env_inst, "last_action_name", "N/A")
                upgrades = getattr(env_inst, "upgrades_purchased_since_retreat", 0)
                advance_locked = getattr(env_inst, "advance_locked", False)
                has_boss = getattr(env_inst, "has_boss_timer_active", False)
                steps_boss = getattr(env_inst, "steps_fighting_current_boss", 0)
                event_msg = getattr(env_inst, "last_event_msg", None)

                boss_tag = f"🔥Boss({steps_boss})" if has_boss else "🌾Farm"
                upg_tag = f"Upg:{upgrades}/5🔒" if advance_locked else f"Upg:{upgrades}🔓"

                zone_str = f"Z:{current_zone}(Max:{max_zone}) | Rew:{last_rew:+.1f}(Tot:{ep_rew:.0f}) | {last_act} | {upg_tag} | {boss_tag}"

                now_str = datetime.datetime.now().strftime("%H:%M:%S")

                # 1. Registrar eventos significativos en consola y archivo de log
                if event_msg:
                    log_line = f"[{now_str} | Paso {self.num_timesteps:<6} | Z:{current_zone} | Rew:{last_rew:+.1f} | Tot:{ep_rew:.1f}] {event_msg}"
                    tqdm.write(log_line)
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(log_line + "\n")
                    with open(self.events_file, "a", encoding="utf-8") as f:
                        record = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "step": self.num_timesteps,
                            "zone": current_zone,
                            "max_zone": max_zone,
                            "action": last_act,
                            "last_reward": last_rew,
                            "total_reward": ep_rew,
                            "upgrades": upgrades,
                            "locked": advance_locked,
                            "boss_active": has_boss,
                            "event": event_msg
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    env_inst.last_event_msg = None

                # 2. Resumen periódico (Heartbeat) cada 250 pasos
                if (self.num_timesteps - self.last_heartbeat_step) >= 250:
                    self.last_heartbeat_step = self.num_timesteps
                    hb_line = f"[{now_str} | Paso {self.num_timesteps:<6}] 📍 Telemetría: Zona {current_zone} (Récord {max_zone}) | Recompensa Ep: {ep_rew:.1f} | Acción reciente: {last_act} | {upg_tag} | {boss_tag}"
                    tqdm.write(hb_line)
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(hb_line + "\n")

            except Exception:
                zone_str = "Zona: N/A"

            if self.pbar is not None:
                self.pbar.set_postfix_str(zone_str)
                self.pbar.update(delta)

            # PODA AUTOMÁTICA (PRUNING)
            if self.trial is not None and (self.num_timesteps - self.last_prune_check) >= self.prune_interval:
                self.last_prune_check = self.num_timesteps
                try:
                    ep_rewards = getattr(venv.envs[0], "get_episode_rewards", lambda: [])()
                    mean_rew = float(np.mean(ep_rewards[-5:])) if ep_rewards else 0.0
                except Exception:
                    mean_rew = 0.0

                intermediate_score = (max_zone * 20.0) + mean_rew
                self.trial.report(intermediate_score, step=self.num_timesteps)

                if self.trial.should_prune():
                    if self.pbar is not None:
                        self.pbar.set_postfix_str(f"✂️ PODADO (Paso {self.num_timesteps})")
                        self.pbar.close()
                    print(f"\n✂️ [PODA ACTIVA] Trial #{self.trial_num} podado tempranamente en el paso {self.num_timesteps} (Zona {max_zone}) por rendimiento inferior a la media.")
                    raise optuna.TrialPruned()

        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()




class PeriodicCheckpointCallback(BaseCallback):
    """
    Guarda automáticamente puntos de control (checkpoints) cada N pasos y mantiene actualizado
    el archivo models/clicker_heroes_latest.zip para permitir reanudación en cualquier momento.
    """
    def __init__(self, save_freq: int = 5000, save_dir: str = "models/checkpoints", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.latest_path = os.path.join("models", "clicker_heroes_latest")
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step_path = os.path.join(self.save_dir, f"clicker_heroes_step_{self.num_timesteps}")
            self.model.save(step_path)
            self.model.save(self.latest_path)
            print(f"\n💾 [CHECKPOINT AUTO-GUARDADO] Paso {self.num_timesteps} -> {self.latest_path}.zip")
        return True


def train_single_agent(
    algo: str,
    timesteps: int,
    save_path: str,
    mock_mode: bool = True,
    resume: bool = False,
    checkpoint_freq: int = 5000,
    lr: float = 0.000593,
    batch_size: int = 128,
    n_filters_base: int = 16,
):
    """Entrena un agente PPO o DQN de extremo a extremo en Clicker Heroes con checkpoints y soporte de reanudación."""
    mode_str = "SIMULADO (Mock)" if mock_mode else "EN VIVO CON CONTROL DE RATON!"
    print(f"\n" + "=" * 70)
    print(f"🚀 [ENTRENAMIENTO REAL] Iniciando agente {algo} en modo {mode_str}")
    print(f"⚙️ Hiperparámetros Óptimos Optuna: LR={lr:.6f} | Batch={batch_size} | Filters={n_filters_base}")
    print(f"💾 Guardado de Checkpoints: Cada {checkpoint_freq} pasos en models/checkpoints/")
    if not mock_mode:
        print("[ALERTA SEGURIDAD] El agente moverá y hará clic dentro de la ventana de Clicker Heroes.")
        print("[PARADA EMERGENCIA] PULSA LA TECLA 'ESC' (1 SOLO TOQUE) PARA DETENER Y GUARDAR EL CHECKPOINT.")
    print("=" * 70 + "\n")

    global_emergency_listener.reset()

    try:
        env = build_stacked_env(mock_mode=mock_mode)
    except WindowNotFoundError as e:
        print(e)
        return

    latest_checkpoint = os.path.join("models", "clicker_heroes_latest.zip")
    latest_no_ext = os.path.join("models", "clicker_heroes_latest")

    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            n_filters_base=n_filters_base,
            depth=3,
        ),
    )

    is_resumed = False
    if resume:
        target_load = save_path if os.path.exists(save_path + ".zip") or os.path.exists(save_path) else latest_checkpoint
        if os.path.exists(target_load) or os.path.exists(target_load + ".zip"):
            print(f"🔄 [REANUDACIÓN ACTIVA] Cargando checkpoint previo desde: {target_load}...")
            if algo.upper() == "PPO":
                model = PPO.load(target_load, env=env, learning_rate=lr, batch_size=batch_size, policy_kwargs=policy_kwargs, verbose=0)
            else:
                model = DQN.load(target_load, env=env, learning_rate=lr, batch_size=batch_size, policy_kwargs=policy_kwargs, verbose=0)
            is_resumed = True
            print("✔ [CHECKPOINT CARGADO] Continuando entrenamiento desde los pesos previos.\n")
        else:
            print(f"⚠️ [AVISO] No se encontró checkpoint previo en '{target_load}'. Iniciando modelo nuevo...\n")

    if not is_resumed:
        if algo.upper() == "PPO":
            model = PPO(
                "CnnPolicy",
                env,
                learning_rate=lr,
                n_steps=256,
                batch_size=batch_size,
                gamma=0.99,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )
        elif algo.upper() == "DQN":
            model = DQN(
                "CnnPolicy",
                env,
                learning_rate=lr,
                buffer_size=10000,
                batch_size=batch_size,
                gamma=0.99,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )
        else:
            raise ValueError(f"Algoritmo no soportado: {algo}")

    safety_callback = EmergencyStopCallback(verbose=0)
    progress_callback = TrialProgressCallback(total_timesteps=timesteps, trial_num=0, total_trials=1)
    checkpoint_callback = PeriodicCheckpointCallback(save_freq=checkpoint_freq, save_dir="models/checkpoints")
    cb_list = CallbackList([safety_callback, progress_callback, checkpoint_callback])

    try:
        model.learn(total_timesteps=timesteps, callback=cb_list, reset_num_timesteps=(not is_resumed))
        if not safety_callback.stop_triggered:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "models", exist_ok=True)
            model.save(save_path)
            model.save(latest_no_ext)
            print(f"\n🎉 [ÉXITO] Entrenamiento completado. Modelo final guardado en: {save_path}.zip y {latest_no_ext}.zip")
    except (EmergencyStopException, KeyboardInterrupt):
        print(f"\n🚨 [PARADA REGISTRADA] Guardando punto de control de emergencia...")
        os.makedirs("models", exist_ok=True)
        model.save(latest_no_ext)
        emergency_file = os.path.join("models", "clicker_heroes_emergency_save")
        model.save(emergency_file)
        print(f"💾 Checkpoint guardado exitosamente en: {latest_no_ext}.zip (y {emergency_file}.zip)")
        print(f"💡 Para reanudar el entrenamiento en cualquier momento, ejecuta con el flag: --resume")
    except Exception as e:
        print(f"\n⚠️ Entrenamiento interrumpido por excepción: {e}")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "models", exist_ok=True)
        model.save(save_path)
        model.save(latest_no_ext)
    finally:
        env.close()



def tune_hyperparameters(algo: str, n_trials: int, timesteps_per_trial: int, mock_mode: bool = True):
    """Ejecuta optimización de hiperparámetros con Optuna."""
    print(f"\n" + "=" * 70)
    print(f"🚀 [AUTORL] INICIANDO OPTIMIZACIÓN BAYESIANA ({n_trials} Ensayos | {timesteps_per_trial} Pasos/Trial)")
    print(f"💡 Pulsa 'ESC' en cualquier momento para detener y guardar el estudio de forma segura.")
    print("=" * 70 + "\n")

    def objective(trial: optuna.Trial) -> float:
        global_emergency_listener.reset()
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        n_filters_base = trial.suggest_categorical("n_filters_base", [16, 32, 64])

        # 1. Reiniciar la partida a un estado base en cada trial para evaluación justa
        if not mock_mode:
            try:
                from src.utils.input_controller import InputController
                from src.utils.window_finder import find_game_window
                reg = find_game_window(title_substring="Clicker Heroes", bring_to_front=True)
                reset_ctrl = InputController(region=reg, mock_mode=False)
                
                save_file_path = os.path.join("models", "base_save.txt")
                if os.path.exists(save_file_path):
                    with open(save_file_path, "r", encoding="utf-8") as f:
                        save_data = f.read().strip()
                    if save_data:
                        reset_ctrl.soft_reset_game_save(save_data)
                    else:
                        print(f"⚠️ El archivo {save_file_path} está vacío. Saltando Soft Reset.")
                else:
                    print(f"⚠️ No se encontró el archivo {save_file_path}.")
                    print("   Debes crear este archivo y pegar dentro tu string de guardado para habilitar el Soft Reset entre trials de Optuna.")
            except Exception as reset_err:
                print(f"[AVISO] Aviso de reinicio en trial {trial.number}: {reset_err}")

        env = build_stacked_env(mock_mode=mock_mode)



        policy_kwargs = dict(
            features_extractor_class=CustomCNNFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                n_filters_base=n_filters_base,
                depth=3,
            ),
        )

        if algo.upper() == "PPO":
            model = PPO(
                "CnnPolicy",
                env,
                learning_rate=lr,
                n_steps=256,
                batch_size=batch_size,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )
        else:
            model = DQN(
                "CnnPolicy",
                env,
                learning_rate=lr,
                batch_size=batch_size,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )

        safety_callback = EmergencyStopCallback(verbose=0)
        progress_callback = TrialProgressCallback(
            total_timesteps=timesteps_per_trial,
            trial_num=trial.number,
            total_trials=n_trials,
            trial=trial,
            prune_interval=500,
        )
        cb_list = CallbackList([safety_callback, progress_callback])

        try:
            model.learn(total_timesteps=timesteps_per_trial, callback=cb_list)

            # Obtener datos de zonas alcanzadas y recompensa acumulada de la partida real
            final_zone = 1
            max_zone = 1
            trial_reward = 0.0
            try:
                venv = env
                while hasattr(venv, "venv"):
                    venv = venv.venv
                env_inst = venv.envs[0].unwrapped
                final_zone = getattr(env_inst, "current_zone", 1)
                max_zone = getattr(env_inst, "max_zone_reached", 1)
                ep_rewards = getattr(venv.envs[0], "get_episode_rewards", lambda: [])()
                trial_reward = float(np.sum(ep_rewards)) if ep_rewards else float(getattr(env_inst, "current_step", 0))
            except Exception:
                pass

            trial.set_user_attr("final_zone", final_zone)
            trial.set_user_attr("max_zone", max_zone)
            trial.set_user_attr("mean_reward", trial_reward)

            pct = ((trial.number + 1) / n_trials) * 100
            score = (max_zone * 100.0) + trial_reward
            print(f"\n✔ [TRIAL #{trial.number} FINALIZADO | Progreso: {trial.number + 1}/{n_trials} ({pct:.0f}%)]")
            print(f"   📊 Recompensa Acumulada: {trial_reward:.2f} (Puntuación Optuna: {score:.2f})")
            print(f"   🏰 Zona Final: Nivel {final_zone} | 👑 Zona Récord Máxima: Nivel {max_zone}")
            print(f"   ⚙️ Hiperparámetros: LR={lr:.6f}, Batch={batch_size}, Filters={n_filters_base}\n")

        except optuna.TrialPruned:
            try:
                venv = env
                while hasattr(venv, "venv"):
                    venv = venv.venv
                env_inst = venv.envs[0].unwrapped
                trial.set_user_attr("final_zone", getattr(env_inst, "current_zone", 1))
                trial.set_user_attr("max_zone", getattr(env_inst, "max_zone_reached", 1))
            except Exception:
                pass
            print(f"✂️ [TRIAL #{trial.number} PODADO] Descartado para ahorrar tiempo y avanzar al siguiente ensayo...\n")
            raise
        finally:
            env.close()
        return score

    pruner = MedianPruner(
        n_startup_trials=2,    # Los primeros 2 ensayos se completan enteros para crear la línea base
        n_warmup_steps=1500,   # Pasos mínimos antes de considerar podar un ensayo
        interval_steps=500     # Evalúa poda cada 500 pasos
    )
    
    # Guardar en base de datos SQLite para no perder el progreso
    study_name = "clicker_heroes_ppo_tuning"
    storage_name = "sqlite:///optuna_study.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize", 
        pruner=pruner,
        load_if_exists=True  # Carga el estudio si ya existía para continuar donde lo dejó
    )
    try:
        study.optimize(objective, n_trials=n_trials)
        print("\n" + "=" * 75)
        print("🎯 [AUTORL] BÚSQUEDA Y OPTIMIZACIÓN COMPLETADA CON ÉXITO")
        print("=" * 75)
        best = study.best_trial
        print(f"⭐ MEJOR TRIAL: #{best.number}")
        print(f"⭐ MEJOR RECOMPENSA MEDIA: {best.user_attrs.get('mean_reward', best.value):.2f}")
        print(f"⭐ ZONA RÉCORD DEL MEJOR TRIAL: Nivel {best.user_attrs.get('max_zone', 'N/A')}")
        print(f"⭐ MEJORES HIPERPARÁMETROS:")
        for k, v in best.params.items():
            print(f"   - {k}: {v}")

        print("\n📋 RESUMEN DETALLADO DE TODOS LOS ENSAYOS (TRIALS):")
        print("-" * 75)
        print(f"{'Trial #':<9} | {'Estado':<10} | {'Recompensa':<12} | {'Zona Final':<11} | {'Zona Récord':<12} | {'Parámetros'}")
        print("-" * 75)
        for t in study.trials:
            fz = t.user_attrs.get('final_zone', 'N/A')
            mz = t.user_attrs.get('max_zone', 'N/A')
            raw_rew = t.user_attrs.get('mean_reward', t.value)
            lr_str = f"{t.params.get('learning_rate', 0):.5f}"
            bs_str = f"{t.params.get('batch_size', 'N/A')}"
            fl_str = f"{t.params.get('n_filters_base', 'N/A')}"
            if t.state == optuna.trial.TrialState.PRUNED:
                print(f"Trial #{t.number:<3} | ✂️ PODADO  | {'N/A':<12} | Zona {fz:<6} | Zona {mz:<7} | LR={lr_str}, Batch={bs_str}, Filters={fl_str}")
            elif t.value is not None:
                val_str = f"{raw_rew:.2f}" if isinstance(raw_rew, (int, float)) else str(raw_rew)
                print(f"Trial #{t.number:<3} | ✔ OK       | {val_str:<12} | Zona {fz:<6} | Zona {mz:<7} | LR={lr_str}, Batch={bs_str}, Filters={fl_str}")
        print("=" * 75 + "\n")
    except (EmergencyStopException, KeyboardInterrupt):
        print("\n🚨 [PARADA DE EMERGENCIA] Búsqueda AutoRL cancelada por el usuario.")





def main():
    parser = argparse.ArgumentParser(description="AutoRL Clicker Heroes Trainer - Fase 3")
    parser.add_argument("--mode", type=str, choices=["train", "tune", "eval"], default="train", help="Modo de ejecución")
    parser.add_argument("--algo", type=str, choices=["PPO", "DQN"], default="PPO", help="Algoritmo RL")
    parser.add_argument("--timesteps", type=int, default=100000, help="Pasos de entrenamiento para la sesión")
    parser.add_argument("--trials", type=int, default=10, help="Número de ensayos para AutoRL (mode=tune)")
    parser.add_argument("--live", action="store_true", help="Activa el modo en vivo con clics reales en pantalla (por defecto es mock)")
    parser.add_argument("--model-path", type=str, default="models/clicker_heroes_latest", help="Ruta del modelo guardado")
    parser.add_argument("--resume", action="store_true", help="Reanuda el entrenamiento desde el último checkpoint guardado")
    parser.add_argument("--checkpoint-freq", type=int, default=5000, help="Frecuencia de guardado de checkpoints (en pasos)")
    parser.add_argument("--lr", type=float, default=0.000593, help="Tasa de aprendizaje (default: óptimo de Optuna 0.000593)")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamaño de lote (default: óptimo de Optuna 128)")
    parser.add_argument("--filters", type=int, default=16, help="Filtros base CNN (default: óptimo de Optuna 16)")
    args = parser.parse_args()

    mock_mode = not args.live

    try:
        if args.mode == "tune":
            tune_hyperparameters(
                algo=args.algo,
                n_trials=args.trials,
                timesteps_per_trial=args.timesteps,
                mock_mode=mock_mode,
            )
        elif args.mode == "train":
            train_single_agent(
                algo=args.algo,
                timesteps=args.timesteps,
                save_path=args.model_path,
                mock_mode=mock_mode,
                resume=args.resume,
                checkpoint_freq=args.checkpoint_freq,
                lr=args.lr,
                batch_size=args.batch_size,
                n_filters_base=args.filters,
            )
        elif args.mode == "eval":
            env = build_stacked_env(mock_mode=mock_mode)
            load_target = args.model_path if os.path.exists(args.model_path + ".zip") or os.path.exists(args.model_path) else os.path.join("models", "clicker_heroes_latest")
            print(f"🔬 [EVALUACIÓN] Cargando modelo desde: {load_target}...")
            if args.algo == "PPO":
                model = PPO.load(load_target, env=env)
            else:
                model = DQN.load(load_target, env=env)

            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
            print(f"--- Evaluación: Recompensa Media = {mean_reward:.2f} +/- {std_reward:.2f} ---")
            env.close()
    except WindowNotFoundError as e:
        print(e)
    except (EmergencyStopException, KeyboardInterrupt):
        print("\n🚨 [PARADA DE EMERGENCIA] Ejecución cancelada limpiamente por el usuario.")



if __name__ == "__main__":
    main()
