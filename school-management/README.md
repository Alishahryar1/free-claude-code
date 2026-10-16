# Gestion scolaire

Application web en français pour suivre les élèves, inscriptions, classes, enseignants et contrats, paiements, dépenses, notes, bulletins, cartes et dossiers individuels. Les calculs financiers et les moyennes sont tirés des écritures saisies. Les listes et documents s’impriment depuis le navigateur.

## Version exécutable Windows

Un paquet Windows autonome se lance depuis `dist\GestionScolaire\GestionScolaire.exe`; Python n’a pas besoin d’être installé sur le PC qui l’utilise. Le paquet ZIP `dist\GestionScolaire-Windows.zip` est prêt à distribuer. Extrayez tout son contenu, puis lancez directement `GestionScolaire.exe`, ou installez l’application dans le profil Windows sans privilèges administrateur en exécutant dans PowerShell :

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Installer-GestionScolaire.ps1
```

L’installateur crée des raccourcis sur le Bureau et dans le menu Démarrer, avec l’icône incluse dans le programme. Python n’a pas besoin d’être installé sur le PC qui l’utilise. Au démarrage, la fenêtre du serveur s’ouvre et l’application apparaît dans le navigateur à <http://127.0.0.1:8765>. Gardez la fenêtre ouverte pendant l’utilisation; fermez-la pour arrêter l’application. Le serveur n’écoute que sur le PC local.

Pour construire ce paquet depuis une copie du dépôt sur Windows, installez [uv](https://docs.astral.sh/uv/), ouvrez PowerShell à la racine du dépôt, puis lancez :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-school-management-windows.ps1
```

Le script génère une icône Windows multi-résolution (16, 24, 32, 48, 64, 128 et 256 px) depuis le fichier source `school-management\assets\school-mark.svg`, puis installe temporairement FastAPI, Uvicorn et PyInstaller dans l’environnement `uv`. L’icône est passée à PyInstaller pour l’intégrer aux ressources de l’EXE; le paquet `onedir` inclut aussi les ressources web. Le résultat est écrit dans `dist\GestionScolaire\`. Construire l’exécutable demande une connexion Internet la première fois pour télécharger ces outils; son fonctionnement quotidien ne demande ni Python, ni `uv`, ni accès réseau externe.

La base, les photos et les sauvegardes restent dans `%LOCALAPPDATA%\FreeClaudeCode\` (base `school-management.sqlite3` et dossier de sauvegardes). Ne déplacez pas la base dans le dossier du programme et protégez l’accès à la session Windows. Copiez le dossier local de sauvegardes vers un support de sauvegarde physique sûr selon la politique de l’établissement.

## Démarrer le serveur local

Depuis la racine du dépôt, avec `uv` installé :

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
uv run --no-project --with "fastapi[standard]" uvicorn free_claude_code.school_management.app:app --host 127.0.0.1 --port 8000
```

Ouvrez <http://127.0.0.1:8000>. L’API n’accepte que les requêtes de la machine locale; elle n’est pas configurée pour un déploiement réseau ou public.

La base SQLite est créée automatiquement dans `%LOCALAPPDATA%\FreeClaudeCode\school-management.sqlite3`. Pour choisir un autre emplacement avant le démarrage :

```powershell
$env:SCHOOL_MANAGEMENT_DB = "$PWD\data\school-management.sqlite3"
```

## Automatisation Python et sauvegardes locales

Le serveur Python est la source de vérité pour les calculs métier et lance un planificateur intégré sans service tiers. Il crée une sauvegarde JSON au démarrage quand des données existent, puis toutes les 24 heures. Les 14 fichiers les plus récents sont gardés; les sauvegardes contiennent les dossiers scolaires, jamais les comptes, hachages de mots de passe ni sessions. Depuis **Paramètres → Sauvegardes automatiques locales**, un administrateur peut consulter l’état et lancer une sauvegarde immédiate. Chaque sauvegarde est écrite atomiquement dans un dossier local.

Par défaut, les fichiers se trouvent dans `school-management-backups` à côté de la base SQLite. Configurez avant le lancement :

```powershell
$env:SCHOOL_MANAGEMENT_BACKUP_DIR = "$PWD\data\sauvegardes"
$env:SCHOOL_MANAGEMENT_BACKUP_INTERVAL_HOURS = "24"
$env:SCHOOL_MANAGEMENT_BACKUP_RETENTION = "14"
# Valeurs acceptées : true/false; false désactive seulement la tâche planifiée.
$env:SCHOOL_MANAGEMENT_AUTO_BACKUP = "true"
```

La fréquence doit être comprise entre 1 et 8760 heures et la rétention entre 1 et 365 sauvegardes. Les erreurs de sauvegarde sont inscrites dans le journal Python et exposées dans l’écran Paramètres. Une sauvegarde automatique n’est pas une copie hors appareil : protégez le PC et copiez périodiquement les fichiers vers un support local sécurisé. Les rapports financiers peuvent aussi être générés en PDF par l’API Python depuis **Comptabilité & rapports**, sur la période choisie; les droits comptables s’appliquent et les détails restent limités aux données accessibles au compte. La clôture annuelle et les décisions de promotion restent une action explicite d’un personnel habilité, car elles nécessitent la validation du bilan et des orientations. Les cartes et bulletins restent imprimables par le navigateur; leur génération PDF planifiée et l’envoi WhatsApp en arrière-plan ne sont pas activés.

**Première connexion**

- Nom d’utilisateur : `Moussa`
- Mot de passe initial : `79408093Ms@`

Le mot de passe est haché côté serveur et n’est jamais réaffiché après la connexion. Depuis **Paramètres → Compte administrateur**, le Proviseur ou l’Administrateur connecté peut modifier son nom d’utilisateur et remplacer son mot de passe en indiquant le mot de passe actuel. Remplacez les identifiants initiaux avant de conserver des données réelles.

## Rôles et permissions

Les rôles par défaut sont disponibles à la création des comptes :

| Rôle | Accès initial |
|---|---|
| Proviseur | Super-administrateur, tous les modules et activation des comptes |
| Administrateur | Accès étendu, activation des comptes; suppression de l’historique financier et d’audit interdite |
| Censeur | Enseignants, contrats, notes, coefficients et bulletins |
| Direction Crèche & Primaire | Élèves, classes, enseignants, notes, présences et suivi de Crèche/Primaire uniquement |
| Direction Secondaire & Lycée | Élèves, classes, enseignants, notes, présences et suivi du Secondaire/Lycée uniquement |
| Surveillant | Élèves, classes et feuilles de présence; pas de création d’inscription |
| Économe | Inscriptions, paiements, dépenses et rapports; historique financier non supprimable |
| Comptable | Paiements, dépenses, comptabilité et rapports financiers |
| Enseignant | Classes, notes et bulletins des classes qui lui sont attribuées |
| Agent d’accueil | Création d’inscriptions et consultation des élèves, sans suppression |

Dans **Comptes & accès**, le Proviseur attribue les rôles; le Proviseur et l’Administrateur peuvent modifier les autorisations CRUD dans la limite de leurs propres droits. La matrice inclut aussi le droit d’activer/désactiver un compte, qui n’est accordé par défaut qu’au Proviseur et à l’Administrateur. Les rôles Direction reçoivent automatiquement un périmètre serveur verrouillé (Crèche/Primaire ou Secondaire/Lycée); les classes peuvent en plus restreindre les comptes à une classe précise. Les journaux d’audit sont append-only.

Pour limiter un compte à des classes particulières, choisissez ces classes dans **Périmètre des classes** de sa fiche. Au moins une classe doit être choisie pour un compte Enseignant. L’API applique le périmètre aux classes, enseignants affectés, élèves, notes, compositions, présences et paiements liés. Les directeurs peuvent avoir plusieurs comptes distincts, sans accès croisé entre cycles.

## Paramètres et identité visuelle

Dans **Paramètres**, modifiez le nom de l’application et les informations de l’établissement. La zone **Logo & palette de couleurs** accepte des fichiers PNG, JPG/JPEG et SVG de 1,5 Mo maximum. Les trois couleurs (principale, secondaire et accent) et le logo sont prévisualisés avant l’enregistrement; ils sont utilisés dans la navigation, les cartes et les bulletins. Les réglages sont enregistrés dans la base SQLite.

La rubrique **Matières & coefficients** propose un référentiel de départ pour chaque niveau et série; le Censeur peut y saisir les coefficients sans recevoir les droits de paramétrage de l’établissement. Tous les coefficients commencent à 1 comme valeur neutre, pas comme barème officiel : choisissez le niveau, adaptez les noms/options et saisissez les coefficients validés par l’établissement, puis enregistrez. Les nouvelles notes et compositions reprennent le coefficient configuré; les calculs pondérés utilisent le coefficient conservé avec chaque note. Les séries `TSEExp` et `TEXP` sont harmonisées sous `TSEExp`; les anciennes fiches restent lisibles.

## Présences

La rubrique **Présences** fournit une feuille d’appel par classe et date, avec les statuts Présent, Absent, Retard et Excusé, une observation facultative, l’identification du saisissant et l’impression de la feuille. Une seule saisie par élève et par date est autorisée; l’API vérifie le lien à l’élève, la date, le statut et le périmètre de classe/cycle.

Le rôle **Économe** enregistre les inscriptions, paiements et sorties de caisse détaillées (cahiers, stylos/bics, craies, tenues scolaires et autres catégories), avec description ou justificatif. Les écritures sont conservées et ne peuvent pas être supprimées par défaut.

## Années scolaires, niveaux et passages

Le référentiel de classes comprend Crèche (Petit, Grand), Primaire (1re à 6e), Secondaire (7e à 9e) et Lycée (10e CM; 11e LL, SES, S; 12e TLL, TAL, TSS, TSE, TSExp, TSEco). La fiche d’une classe associe son cycle/niveau et son professeur; le Censeur peut aussi gérer les affectations depuis la fiche enseignant.

Les listes par classe sont produites à partir des inscriptions propres à chaque année. La rubrique **Classes** permet de consulter une année passée et affiche le bilan annuel : moyenne pondérée par coefficient, décision de passage selon le seuil (10/20 par défaut, modifiable dans Paramètres), redoublement si le seuil n’est pas atteint et fin de cycle en Terminale. La clôture est une action explicite qui vérifie les notes et les classes de destination, conserve les inscriptions et décisions de l’année fermée, puis active l’année suivante. Créez les classes manquantes du niveau suivant avant la clôture; pour une orientation en filière, le niveau cible est choisi parmi les classes configurées. Une année déjà clôturée ne peut pas l’être une seconde fois depuis l’application.

## WhatsApp des listes de classe

Renseignez le téléphone du professeur et affectez-le à une classe. Par défaut, le mode **Test** prépare et journalise le texte sans ouvrir WhatsApp ni envoyer de message. Le mode **Lien WhatsApp** fabrique un lien `wa.me` avec le contact validé et la liste annuelle préremplie; l’utilisateur ouvre le lien, vérifie le destinataire et confirme l’envoi dans WhatsApp. Il n’existe pas d’envoi automatique en arrière-plan ni de jeton/API WhatsApp préconfiguré. Cette intégration n’invente ni ne stocke d’identifiants; l’envoi via l’API officielle nécessiterait une configuration Meta/WhatsApp et des secrets serveur distincts avant toute activation.

## Modèle imprimable de carte scolaire

Dans **Paramètres → Modèle de carte scolaire**, choisissez la disposition horizontale/verticale et les champs facultatifs (photo, matricule, date de naissance, responsable et téléphone familial). La photo téléversée depuis la fiche Élève est stockée dans la base locale, apparaît sur la carte imprimable si l’option Photo est activée, et est remplacée par les initiales lisibles si aucune photo n’est fournie. Les photos JPEG, PNG ou WebP sont limitées à 1 Mo avant traitement; elles sont redimensionnées localement à 640 × 800 pixels maximum et compressées avant enregistrement. L’aperçu du modèle se met à jour pendant la sélection; l’enregistrement conserve le modèle dans SQLite. Le logo d’établissement configuré et le drapeau du Mali embarqué localement restent présents sur chaque carte et bulletin. Les deux documents restent disponibles via **Imprimer**; ils ne dépendent d’aucune ressource distante pour le drapeau.

Les actions sensibles (connexion, changements de compte et de permissions, activation, modifications de données et réinitialisation) apparaissent dans le **Journal d’audit**. Il ne propose aucune action de suppression.

## Rapports par période, évaluations et conduite

Dans **Comptabilité & rapports**, renseignez les dates de début et de fin pour filtrer les recettes, dépenses, solde net et grand livre imprimable. Le crédit des familles correspond au solde actuel, distinct des mouvements filtrés. Effacez les dates pour retrouver toutes les périodes.

Dans **Notes & bulletins**, sélectionnez l’année et le trimestre. Les années précédentes sont consultables sans modification. Le bouton **Imprimer le suivi annuel** produit le dossier d’évaluation de l’élève, regroupé par année et classe, avec devoirs/compositions, notes, coefficients, moyennes par matière, appréciations et progression d’une année à l’autre.

Le **capital initial de conduite** vaut 18 points et peut être réglé à 18 ou 20 dans Paramètres. Le registre **Conduite & incidents** exige un retrait saisi manuellement, un motif et une date; aucun barème automatique par infraction n’est inventé. Un incident reste en attente jusqu’à validation par un compte ayant le droit de modifier la Conduite. Seuls les retraits validés affectent la note du bulletin et du dossier élève; le solde ne descend jamais sous zéro. Les validations sont auditées et les incidents validés deviennent immuables. Avec le serveur FastAPI, l’audit et la validation sont protégés côté serveur. Le mode démonstration local n’offre pas le même journal d’audit serveur.

## Données de démonstration et limites

Les données scolaires initiales sont synthétiques. La réinitialisation des données de démonstration est réservée au Proviseur; elle conserve les comptes et le journal d’audit. Le logo et la palette fournis sont des exemples modifiables. **Paramètres → Gestion des données** permet de télécharger puis restaurer un fichier JSON local; la restauration valide la forme des registres, conserve les comptes, les hachages de mots de passe et l’audit, et n’importe jamais de compte depuis le fichier.

Le stockage est exclusivement local au PC de l’établissement : en mode serveur, les données et photos (encodées dans le registre SQLite local) ne sont pas recopiées dans le stockage du navigateur; en mode démonstration, elles restent dans le localStorage de ce navigateur. Les exports et sauvegardes locales incluent les photos pour permettre leur restauration et n’incluent jamais les comptes ou mots de passe; la limite de sauvegarde est de 64 Mo. Aucun fournisseur externe ni synchronisation cloud n’est activé. L’ouverture manuelle d’un lien WhatsApp prérempli reste une action distincte déclenchée par le personnel. Conservez les sauvegardes sur un support local protégé et vérifiez-les régulièrement.

Le serveur local n’est pas un déploiement multi-utilisateur : il n’expose volontairement pas l’API au réseau. Pour un usage en établissement, il faudra définir l’hébergement, HTTPS, les sauvegardes, les obligations locales de protection des données et les formats officiels de documents.

## Tests ciblés

Avec l’environnement de développement du dépôt installé :

```powershell
python -m pytest tests\test_school_management.py -n 0 -q
```
