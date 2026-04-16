"""
MediSign Assist — Liste centralisée des signes LSF cibles
==========================================================
Source unique de vérité pour les 20 signes médicaux d'urgence.
Importer depuis tous les notebooks et scripts.

Catégories :
    Communication : OUI, NON
    Urgence       : APPELER, VITE, AIDER, MORT
    Besoin vital  : MANGER, BOIRE, DORMIR
    Mobilité      : MARCHER, TOMBER
    Symptôme      : CHAUD, FROID, SOUFFRIR, PLEURER
    État          : PEUR, 
    Corps         : TETE
    Médical       : HOPITAL, MEDECIN
"""

SIGNS_TARGET = [
    "OUI",      "NON",      "APPELER",  "VITE",     "MANGER",
    "COMPRENDRE",    "BOIRE",    "MALADE.VENTRE",  "CHAUD",    "DORMIR",
    "RESPIRER",   "TOMBER",  "FROID",    "FAIBLE",     "ENCEINTE",
    "HOPITAL",  "SOUFFRIR", "PAS.SOUFFLER",     "FORT",     "MEDECIN",
]

SIGN_TO_IDX = {s: i for i, s in enumerate(SIGNS_TARGET)}
IDX_TO_SIGN = {i: s for i, s in enumerate(SIGNS_TARGET)}
NUM_CLASSES = len(SIGNS_TARGET)

