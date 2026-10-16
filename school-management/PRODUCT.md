# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Équipe de direction et personnel administratif d’un complexe scolaire, travaillant depuis un navigateur pour suivre les inscriptions, les classes, les paiements, les dépenses et les résultats scolaires.

## Product Purpose

Réunir les opérations administratives et scolaires courantes d’un établissement dans une interface en français. Le succès signifie retrouver et mettre à jour les dossiers et les totaux sans multiplier les registres papier.

## Positioning

Un registre scolaire unique où les inscriptions, la situation financière et les bulletins d’une même classe restent cohérents.

## Operating Context

Utilisation sur ordinateur ou téléphone par le personnel scolaire. Les fonctions demandées comprennent l’impression des listes d’élèves, cartes scolaires et bulletins, ainsi que le suivi de paiements en francs CFA. Les niveaux couvrent Crèche à Terminale; les inscriptions sont historisées par année et les passages se calculent sur la moyenne annuelle pondérée avec un seuil configurable. Les professeurs sont affectés à des classes et les listes WhatsApp sont préparées de manière contrôlée, avec un mode test sans envoi.

## Capabilities and Constraints

L’application sert une interface web depuis FastAPI et conserve ses données dans SQLite sur la machine locale. Les mots de passe sont hachés côté serveur; les permissions CRUD, les périmètres de classe, les sessions, les clôtures annuelles et les préparations WhatsApp sont contrôlés par l’API et journalisés. Le partage WhatsApp utilise un lien prérempli à confirmer par un humain; aucun envoi via API Meta n’est configuré. Le drapeau du Mali est un SVG local embarqué dans les documents. Le serveur n’écoute que sur l’interface loopback par défaut; ce produit n’est pas encore configuré pour un déploiement réseau ou multi-utilisateur. Les données d’exemple et le logo d’école sont synthétiques et modifiables. Les formats officiels des documents scolaires restent à confirmer par l’établissement.

## Brand Commitments

Nom de travail : Gestion scolaire. Interface en français.

## Evidence on Hand

Le besoin fonctionnel fourni par l’utilisateur. Aucun logo d’établissement, dossier réel, modèle de bulletin ou liste de tarifs n’a été fourni. Toutes les données d’exemple et le logo inclus doivent être clairement fictifs.

## Product Principles

- Garder les dossiers d’élèves, les classes et les écritures financières reliés.
- Afficher les totaux financiers à partir des enregistrements, plutôt que de les modifier séparément.
- Rendre les droits de chaque compte compréhensibles par module et action.
- Préserver l’impression des documents scolaires usuels.
