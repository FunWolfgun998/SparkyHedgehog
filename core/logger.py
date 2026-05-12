import os
import time
import config


class PrintManager:
    def __init__(self):
        self.console_enabled = True  # Semplice booleano locale

        self.file_enabled = getattr(config, 'SAVE_TEXT_LOGS', False)
        self.log_path = None

        if self.file_enabled:
            log_dir = os.path.join(config.LOG_DIR, "text_logs")
            os.makedirs(log_dir, exist_ok=True)
            self.log_path = os.path.join(log_dir, f"{config.CURRENT_RUN_NAME}_debug.log")

            # Crea il file solo se non esiste
            if not os.path.exists(self.log_path):
                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.write(f"--- INIZIO LOG RUN: {config.CURRENT_RUN_NAME} ---\n")

    def log(self, message: str, **kwargs):
        if not self.console_enabled and not self.file_enabled:
            return  # Esci subito per risparmiare calcoli se è tutto spento

        try:
            formatted_msg = message.format(**kwargs) if kwargs else message
        except KeyError as e:
            formatted_msg = f"{message}[Errore Formato: Manca {e}]"

        timestamp = time.strftime("%H:%M:%S")
        final_msg = f"[{timestamp}] {formatted_msg}"

        if self.console_enabled:
            print(final_msg, flush=True)

        if self.file_enabled and self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(final_msg + "\n")


# Istanza globale per questo specifico processo
sparky_logger = PrintManager()