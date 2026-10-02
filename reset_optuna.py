"""
Script de Limpieza y Reinicio de Historial de Optuna (reset_optuna.py)
Elimina bases de datos locales (.db), estudios en SQLite y cachés previas
para que la próxima optimización arranque desde el Trial 0 sin sesgos.

Uso:
    python reset_optuna.py
"""
import os
import sys
import glob
import optuna

# Configurar encoding UTF-8 seguro para Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def reset_optuna_studies():
    print("=" * 70)
    print("[AUTORL RESET] Limpiando historial y bases de datos de Optuna...")
    print("=" * 70)


    # 1. Buscar y eliminar archivos SQLite .db locales
    db_patterns = ["*.db", "optuna*.db", "models/*.db"]
    deleted_files = 0

    for pattern in db_patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
                print(f"🗑️ Base de datos eliminada: {filepath}")
                deleted_files += 1
            except Exception as e:
                print(f"⚠️ No se pudo eliminar {filepath}: {e}")

    # 2. Intentar borrar estudios por nombre si existe storage SQLite común
    common_db_url = "sqlite:///optuna_clicker_heroes.db"
    study_names = ["clicker_heroes_ppo_tuning", "clicker_heroes_study", "autorl_clicker_heroes"]

    for name in study_names:
        try:
            optuna.delete_study(study_name=name, storage=common_db_url)
            print(f"✔ Estudio '{name}' borrado de SQLite.")
        except Exception:
            pass

    if deleted_files == 0:
        print("✔ No se encontraron bases de datos residuales. El estudio está listo y limpio para arrancar desde Trial 0.")
    else:
        print(f"✔ Se han limpiado {deleted_files} archivo(s) de base de datos correctamente.")

    print("\n🚀 ¡Listo! El próximo 'python train_clicker_heroes_autorl.py --mode tune' arrancará desde el Trial #0.")
    print("=" * 70)


if __name__ == "__main__":
    reset_optuna_studies()
