from typing import override  # <-- Ajoutez cette ligne
from lib.PluginBase import PluginBase, PluginManifest
from lib.PluginHelper import PluginHelper, STTModel
from lib.PluginSettingDefinitions import (
    PluginSettings,
    SettingsGrid,
    TextSetting,
    ModelProviderDefinition,
)
from lib.Logger import log
from speech_recognition import AudioData  # Importez AudioData depuis speech_recognition
import requests
import io
import sys
import os

# Ajoutez le dossier 'deps' au chemin de recherche des modules Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deps"))

# --- Modèle STT pour Voxtral ---
class MistralVoxtralSTTModel(STTModel):
    def __init__(self, api_key: str, model: str = "voxtral-mini-latest"):
        super().__init__("mistral_voxtral")
        self.api_key = api_key
        self.model = model
        # Endpoint officiel pour Voxtral Transcribe (juin 2026)
        #self.api_url = "https://api.mistral.ai/api/v1/transcribe"
        self.api_url = "https://api.mistral.ai/v1/audio/transcriptions"

    def transcribe(self, audio: bytes | AudioData) -> str:
        try:
            # Convertir AudioData en bytes WAV si nécessaire
            if isinstance(audio, AudioData):
                audio_bytes = audio.get_wav_data()
            else:
                audio_bytes = audio

            # Préparer la requête multipart/form-data
            files = {
                "file": ("audio.wav", audio_bytes, "audio/wav"),
            }
            data = {
                "model": self.model,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()

            # Extraire le texte de la réponse
            result = response.json()
            if isinstance(result, dict):
                if "text" in result:
                    return result["text"]
                elif "output" in result:
                    return result["output"]
            return str(result)

        except requests.exceptions.HTTPError as e:
            log("error", f"Erreur HTTP {e.response.status_code}: {e.response.text}")
            return ""
        except Exception as e:
            log("error", f"Erreur de transcription: {e}")
            return ""
# --- Plugin principal ---
class MistralVoxtralSTTPlugin(PluginBase):
    def __init__(self, plugin_manifest: PluginManifest):
        super().__init__(plugin_manifest)
        # Définir les paramètres du plugin (affichés dans l'UI)
        self.settings_config = PluginSettings(
            key="MistralVoxtralSTT",
            label="Mistral Voxtral STT",
            icon="mic",
            grids=[
                SettingsGrid(
                    key="general",
                    label="Configuration",
                    fields=[
                        TextSetting(
                            key="api_key",
                            label="Clé API Mistral",
                            type="text",
                            placeholder="Entrez votre clé API Mistral",
                            default_value="",
                        ),
                        TextSetting(
                            key="model",
                            label="Modèle Voxtral",
                            type="text",
                            placeholder="voxtral-mini-latest",
                            default_value="voxtral-mini-latest",
                        ),
                    ],
                ),
            ],
        )
        # Définir les fournisseurs de modèles (STT)
        self.model_providers = [
            ModelProviderDefinition(
                kind="stt",
                id="mistral_voxtral",
                label="Mistral Voxtral STT",
            ),
        ]

    @override  # <-- Maintenant valide grâce à l'import
    def on_chat_start(self, helper: PluginHelper):
        """Appelé au démarrage du chat."""
        log("info", "Plugin Mistral Voxtral STT démarré")

    @override  # <-- Maintenant valide grâce à l'import
    def on_chat_stop(self, helper: PluginHelper):
        """Appelé à l'arrêt du chat."""
        log("info", "Plugin Mistral Voxtral STT arrêté")

    def create_model(self, provider_id: str, settings: dict) -> STTModel:
        """Crée une instance du modèle STT."""
        if provider_id == "mistral_voxtral":
            api_key = settings.get("api_key", "")
            model = settings.get("model", "voxtral-mini-latest")
            return MistralVoxtralSTTModel(api_key=api_key, model=model)
        raise ValueError(f"Fournisseur STT inconnu: {provider_id}")
