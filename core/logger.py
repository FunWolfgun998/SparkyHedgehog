import os
import time
import config

class PrintManager:
    def __init__(self):
        # Stato della console (di default su True, ma disattivabile)
        self.console_enabled = True
        
        # Stato del file di log (letto dal config)
        self.file_enabled = getattr(config, 'SAVE_TEXT_LOGS', False)
        self.log_file = None

        if self.file_enabled:
            log_dir = os.path.join(config.LOG_DIR, "text_logs")
            os.makedirs(log_dir, exist_ok=True)
            # Crea un file univoco per questa run (es: Round_11_debug.log)
            file_path = os.path.join(log_dir, f"{config.CURRENT_RUN_NAME}_debug.log")
            self.log_file = open(file_path, "a", encoding="utf-8")
            self._write_to_file(f"--- INIZIO LOG RUN: {config.CURRENT_RUN_NAME} ---")

    def log(self, message: str, **kwargs):
        """
        Stampa e/o salva un messaggio. 
        Usa i kwargs per formattare: log("Hit! {dmg}", dmg=10)
        """
        # Formattazione dinamica modulare
        try:
            formatted_msg = message.format(**kwargs) if kwargs else message
        except KeyError as e:
            formatted_msg = f"{message}[Errore Formato: Manca {e}]"

        timestamp = time.strftime("%H:%M:%S")
        final_msg = f"[{timestamp}] {formatted_msg}"

        # 1. Stampa su console SOLO se non siamo in modalità performance
        if self.console_enabled:
            print(final_msg, flush=True)

        # 2. Salva su disco SEMPRE (se abilitato da config)
        if self.file_enabled:
            self._write_to_file(final_msg)

    def _write_to_file(self, msg: str):
        if self.log_file and not self.log_file.closed:
            self.log_file.write(msg + "\n")
            self.log_file.flush() # Forza la scrittura su disco immediata

    def toggle_performance_mode(self):
        """Inverte lo stato della stampa a schermo."""
        self.console_enabled = not self.console_enabled
        stato = "ATTIVI" if self.console_enabled else "DISABILITATI (Performance Mode 🚀)"
        
        avviso = f"\n⚙️ [PRINT MANAGER] Log a schermo: {stato}\n"
        print(avviso) # Stampiamo sempre questo avviso per notificare l'utente
        if self.file_enabled:
            self._write_to_file(avviso.strip())

    def __del__(self):
        """Chiusura sicura del file a fine training."""
        if self.log_file and not self.log_file.closed:
            self._write_to_file("--- FINE LOG ---")
            self.log_file.close()

# Istanza globale da importare negli altri file
sparky_logger = PrintManager()