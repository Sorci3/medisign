"""
MediSign Assist — Liste centralisée des signes LSF cibles
==========================================================
Source unique de vérité pour les 20 signes médicaux d'urgence.
Importer depuis tous les notebooks et scripts.

Catégories :
    Communication : OUI, NON, COMPRENDRE
    Urgence       : APPELER, VITE
    Besoin vital  : MANGER, BOIRE, DORMIR
    Mobilité      : TOMBER
    Symptôme      : CHAUD, FROID, SOUFFRIR, FORT, FAIBLE, MALADE.VENTRE,
                    PAS.SOUFFLER, RESPIRER, ENCEINTE
    Médical       : HOPITAL, MEDECIN
"""

# L'ordre de cette liste définit les indices de classe (voir SIGN_TO_IDX) et les
# poids .pth entraînés en dépendent : réordonner mélange silencieusement les
# labels des modèles existants. Ajouter uniquement à la fin, et réentraîner.
SIGNS_TARGET = [
    "OUI",      "NON",      "APPELER",  "VITE",     "MANGER",
    "COMPRENDRE",    "BOIRE",    "MALADE.VENTRE",  "CHAUD",    "DORMIR",
    "RESPIRER",   "TOMBER",  "FROID",    "FAIBLE",     "ENCEINTE",
    "HOPITAL",  "SOUFFRIR", "PAS.SOUFFLER",     "FORT",     "MEDECIN",
]

SIGN_TO_IDX = {s: i for i, s in enumerate(SIGNS_TARGET)}
IDX_TO_SIGN = {i: s for i, s in enumerate(SIGNS_TARGET)}
NUM_CLASSES = len(SIGNS_TARGET)

# Un doublon dans SIGNS_TARGET serait absorbé par SIGN_TO_IDX sans erreur.
assert len(SIGN_TO_IDX) == NUM_CLASSES == 20, "SIGNS_TARGET : doublon ou nombre de signes != 20"

