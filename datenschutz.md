---
layout: default
title: Datenschutz
permalink: /datenschutz
---

# Datenschutzerklärung für Custom GPT „Memory-GPT“

## Allgemeines

Dieses GPT ermöglicht es Nutzern, Texte in einer öffentlich zugänglichen Vektor-Datenbank (Pinecone) zu speichern, ähnliche Inhalte abzufragen oder gespeicherte Einträge zu löschen. Die Datenbank ist über eine offene API angebunden und nicht an einzelne Nutzerkonten gebunden.

Die Nutzung dieses GPTs erfolgt freiwillig und anonym. Es findet keine Authentifizierung oder Nutzeridentifikation statt.

---

## Welche Daten werden verarbeitet?

- Texte, die von Nutzern über das GPT eingegeben werden
- Embeddings (Vektorrepräsentationen) dieser Texte, erzeugt über OpenAI
- Speicherung dieser Daten in einem gemeinsam genutzten Pinecone-Index

---

## Wer kann auf die Daten zugreifen?

Alle Nutzer, die dieses GPT verwenden, haben technischen Zugriff auf **alle gespeicherten Inhalte** in der Datenbank. Es gibt **keine Zugriffsbeschränkung** zwischen einzelnen Nutzern.

Das bedeutet:

- Inhalte, die von einem Nutzer gespeichert werden, können von anderen Nutzern über die Suchfunktion (Similarity Search) gefunden und gelesen werden.
- Eine Unterscheidung nach Nutzeridentität findet nicht statt.
- Daten bleiben in der Datenbank, **bis sie aktiv durch einen Nutzer gelöscht oder die gesamte Datenbank geleert wird**.

---

## Keine Speicherung personenbezogener Daten

Dieses GPT ist **nicht dafür vorgesehen, personenbezogene Daten zu speichern oder zu verarbeiten**. Nutzer werden ausdrücklich aufgefordert, **keine sensiblen oder identifizierbaren Informationen** einzugeben.

Für sämtliche Inhalte sind allein die Nutzer verantwortlich.

---

## Dienste Dritter

- **OpenAI, Inc.** – zur Erzeugung von Embeddings: https://openai.com/privacy
- **Pinecone Systems, Inc.** – Vektor-Datenbankanbieter: https://www.pinecone.io/privacy/

---

## Speicherort und Dauer

Die gespeicherten Daten (Text + Embedding) verbleiben im Pinecone-Index, **bis sie manuell gelöscht werden**. Es erfolgt keine automatische Löschung, keine Zeitbegrenzung und keine systematische Verwaltung durch den Betreiber.

---

## Haftungsausschluss

Der Betreiber dieses GPTs übernimmt keine Haftung für Inhalte, die durch Dritte eingegeben und öffentlich zugänglich gemacht werden. Die Plattform dient ausschließlich Demonstrations- und Entwicklungszwecken.

---

## Kontakt

Für Fragen oder Anliegen zur Datenverarbeitung wenden Sie sich an:

📧 [designamo@outlook.de]

Stand: Juli 2025
