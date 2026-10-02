"""
Script Probador Interactivo y Diagnóstico en Vivo para Clicker Heroes (AutoRL)
Permite probar en tiempo real la detección de zonas (OCR), retroceso de jefe (Acción 4),
avance de nivel (Acción 5), compra de héroes y habilidades sin lanzar un entrenamiento largo.
"""
import time
import sys
import queue
import cv2
import numpy as np
from pynput import keyboard

from src.environments.clicker_heroes_env import ClickerHeroesEnv
from src.utils.input_controller import InputController
from src.utils.smart_vision import SmartVisionController
from src.utils.window_finder import find_game_window


def main():
    print("=" * 70)
    print("🎮 [PROBADOR INTERACTIVO EN VIVO] CLICKER HEROES")
    print("=" * 70)
    print("Buscando ventana de Clicker Heroes...")

    try:
        region = find_game_window(title_substring="Clicker Heroes", bring_to_front=True)
    except Exception as e:
        print(f"❌ Error al detectar ventana: {e}")
        return

    print(f"✔ Ventana detectada en: {region}")
    print("\nInicializando entorno en modo EN VIVO...")
    env = ClickerHeroesEnv(
        region=region,
        target_size=(84, 84),
        max_steps=500,
        action_delay=0.04,
        mock_mode=False,
    )
    obs, info = env.reset()

    vision = SmartVisionController()
    event_queue = queue.Queue()

    print("\n" + "=" * 70)
    print("🕹️ CONTROLES DEL PROBADOR (Pulsa teclas para probar en vivo):")
    print("   [0] -> Probar Ataque a Monstruo (Acción 0)")
    print("   [1] -> Probar Subir Nivel Héroe (+NV) (Acción 1)")
    print("   [2] -> Probar Comprar Habilidad Héroe (Acción 2)")
    print("   [3] -> Probar Habilidad Global H (Acción 3)")
    print("   [4] -> Probar RETROCEDER en Jefe / Farm Mode (Acción 4)")
    print("   [5] -> Probar AVANZAR de Nivel / Progression (Acción 5)")
    print("   [ESPACIO] -> Ejecutar 20 pasos automáticos de simulación")
    print("   [ESC] / [Q] -> Salir del probador")
    print("=" * 70 + "\n")

    running = True

    def on_press(key):
        nonlocal running
        try:
            char = key.char.lower() if hasattr(key, 'char') and key.char else ''
            if key == keyboard.Key.esc or char == 'q':
                event_queue.put('quit')
                return False
            elif key == keyboard.Key.space:
                event_queue.put('space')
            elif char in ('0', '1', '2', '3', '4', '5'):
                event_queue.put(char)
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    last_diag_time = time.time()
    try:
        while running:
            # Procesar acciones en el hilo principal (100% seguro para mss / DirectX)
            while not event_queue.empty():
                evt = event_queue.get_nowait()
                if evt == 'quit':
                    print("\n🛑 Saliendo del probador...")
                    running = False
                    break
                elif evt == 'space':
                    print("\n🤖 [AUTO] Ejecutando 20 pasos automáticos en el juego...")
                    for i in range(20):
                        act = InputController.ACTION_MONSTER_CLICK if i % 2 == 0 else (i % 6)
                        _, rew, _, _, step_inf = env.step(act)
                        print(f"   Paso {i+1}/20 | Acción: {act} | Recompensa: {rew:+.2f} | Zona: {step_inf.get('current_zone')}")
                        time.sleep(0.06)
                else:
                    act_map = {
                        '0': InputController.ACTION_MONSTER_CLICK,
                        '1': InputController.ACTION_SMART_UPGRADE,
                        '2': InputController.ACTION_SMART_HERO_SKILLS,
                        '3': InputController.ACTION_GLOBAL_SKILLS_BAR,
                        '4': InputController.ACTION_PREV_ZONE_FARM,
                        '5': InputController.ACTION_NEXT_ZONE_PROGRESS,
                    }
                    action_to_run = act_map.get(evt)
                    if action_to_run is not None:
                        _, rew, _, _, step_inf = env.step(action_to_run)
                        boss_str = "🔥 JEFE ACTIVO" if env.smart_vision.detect_boss_timer(env.last_raw_rgb) else "🌲 Normal"
                        zone_num = step_inf.get('current_zone', 'N/A')
                        print(f"\n⚡ [ACCIÓN {action_to_run}] -> Recompensa: {rew:+.2f} | Zona OCR: {zone_num} | Estado: {boss_str} | Bloqueada: {step_inf.get('blocked_action')}")

            if not running:
                break

            # Diagnóstico continuo cada 1.5 segundos en el hilo principal
            if time.time() - last_diag_time >= 1.5:
                last_diag_time = time.time()
                raw_rgb = env.screen_capture.capture_raw_rgb()
                env.last_raw_rgb = raw_rgb

                detected_zone = vision.detect_zone_number_from_screen(raw_rgb)
                has_boss = vision.detect_boss_timer(raw_rgb)
                hp = env.reward_extractor.extract_hp(raw_rgb)
                ready_skills = vision.detect_ready_global_skills(raw_rgb)
                buyable_btns = vision.detect_buyable_upgrade_buttons(raw_rgb)

                boss_txt = "🔥 SÍ (Temporizador activo)" if has_boss else "❌ No"
                zone_txt = f"Nv. {detected_zone}" if detected_zone is not None else f"Nv. {env.current_zone}"
                skills_txt = f"{len(ready_skills)} activas" if ready_skills else "0"
                btns_txt = f"{len(buyable_btns)} disponibles" if buyable_btns else "0"
                lock_txt = f"🔒 {env.upgrades_purchased_since_retreat}/5 mejoras" if env.advance_locked else "🔓 Listo"

                sys.stdout.write(f"\r📊 [ESTADO EN PANTALLA] Zona: {zone_txt:<8} | Boss: {boss_txt:<24} | Reintento Boss: {lock_txt:<16} | HP: {hp*100:4.1f}% | Habilidades H: {skills_txt:<10} | +NV: {btns_txt:<14}")
                sys.stdout.flush()



            time.sleep(0.04)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        env.close()
        print("\n✔ Probador finalizado correctamente.")


if __name__ == "__main__":
    main()

