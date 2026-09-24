"""
Step 2 - Build the document collection (corpus) and the question set.

Output (in the same folder as this script):
  docs/*.pdf        12 PDF handbooks of the fictional "Klinikum Musterstadt"
  pages.jsonl       one line per content page:  {"id", "doc", "page", "title", "text"}
  questions.jsonl   one line per question:      {"id", "question", "type", "split", "qrels"}

Run:  python data/build_corpus.py
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).parent
FIRST_PAGE = 3  # page 1 = title page, page 2 = table of contents

# =====================================================================================
# PART A - Ten system handbooks built from the same 10-page structure.
# Real enterprise documentation looks like this: every system has an "access" page,
# a "password" page, an "outage" page ... with similar wording but different facts.
# These near-identical pages are the DISTRACTORS that make retrieval hard.
# =====================================================================================

ASPECTS = [
    "Überblick",
    "Zugriff beantragen",
    "Berechtigungen ändern und entziehen",
    "Anmeldung",
    "Zugangsdaten zurücksetzen",
    "Störungen melden",
    "Notfallbetrieb bei Ausfall",
    "Schulung und Ansprechpartner",
    "Datenschutz und Protokollierung",
    "Häufige Fragen",
]

PAGE_TEMPLATES = [
    "{subject} {verb} im Klinikum Musterstadt für {purpose} genutzt. Fachlich verantwortlich ist "
    "{owner}, die technische Betreuung liegt beim Team {team}. {overview_extra}",

    "Der Zugriff auf {akk} wird rollenbasiert über Gruppen im Active Directory ({group}) vergeben. "
    "Den Antrag stellt die oder der Vorgesetzte im Self-Service-Portal mit dem Formular „{form}“. "
    "Verfügbare Rollen sind {roles}. Genehmigt wird der Antrag durch {approver}. Nach der Genehmigung "
    "ist der Zugang in der Regel innerhalb von {days} freigeschaltet.",

    "Ändert sich die Aufgabe einer Person, etwa durch einen Wechsel der Station oder Abteilung, beantragt "
    "die oder der Vorgesetzte die Anpassung der {name}-Rolle ebenfalls im Self-Service-Portal. Nicht mehr "
    "benötigte Rechte werden dabei entzogen. Beim Austritt entfernt die IT alle {name}-Rechte automatisch "
    "am letzten Arbeitstag. Zusätzlich bestätigen die Vorgesetzten {recert} in einer Rezertifizierung, "
    "dass die Berechtigungen ihrer Mitarbeitenden noch benötigt werden.",

    "{login}",

    "{reset}",

    "Bei Problemen {im} hilft häufig zunächst: {first_aid}. Besteht das Problem weiter, wird es beim "
    "IT-Servicedesk unter der Durchwahl 4444 oder im Self-Service-Portal gemeldet. Anzugeben sind die "
    "Fehlermeldung, der Arbeitsplatz und ob die Patientenversorgung betroffen ist. Störungen, die "
    "{prio_condition}, werden mit Priorität {prio} bearbeitet.",

    "{fallback} Nach der Wiederherstellung {recovery}",

    "Neue Mitarbeitende {training}. Ansprechpartner für fachliche Fragen sind die Key-User {keyuser}. "
    "Die Schulungsunterlagen liegen im Intranet unter „{intranet}“.",

    "Alle Anmeldungen und Zugriffe {im} werden protokolliert. Die Protokolle werden {log} aufbewahrt und "
    "nur bei begründetem Verdacht unter Beteiligung von Datenschutz und Personalrat ausgewertet. {privacy}",

    None,  # "Häufige Fragen" is written by hand for every system (see "faq" below)
]

# Four question templates per aspect, each tagged with its style.
# Templates 0-1 are used ONLY for test questions, templates 2-3 ONLY for training questions.
# So a model trained in Step 8 never sees the wording of a test question -> no leakage.
QUESTION_TEMPLATES = [
    [("Wofür {verb} {akk} im Klinikum genutzt?", "close"),
     ("Wer ist fachlich für {akk} verantwortlich?", "paraphrase"),
     ("was ist eigentlich {name}?", "colloquial"),
     ("Welches Team kümmert sich technisch um {akk}?", "paraphrase")],
    [("Wie beantrage ich Zugriff auf {akk}?", "close"),
     ("brauche {name}-Zugang, wie geht das?", "colloquial"),
     ("Wer genehmigt neue Berechtigungen {im}?", "paraphrase"),
     ("Wie lange dauert es, bis ich {akk} nutzen kann?", "paraphrase")],
    [("Ich wechsle die Abteilung – was passiert mit meinen {name}-Rechten?", "paraphrase"),
     ("Wie werden {name}-Berechtigungen entzogen?", "close"),
     ("Wie oft werden die {name}-Berechtigungen überprüft?", "paraphrase"),
     ("Kollege hat gekündigt, hat der noch {name}-Rechte?", "colloquial")],
    [("Wie funktioniert die Anmeldung {im}?", "close"),
     ("wie komm ich in {akk} rein?", "colloquial"),
     ("Brauche ich für {akk} eine eigene Anmeldung?", "paraphrase"),
     ("{name} Login – wie geht das?", "colloquial")],
    [("{name} Passwort vergessen, was tun?", "colloquial"),
     ("Wer hilft mir, wenn ich {im} gesperrt bin?", "paraphrase"),
     ("Wie setze ich meine Zugangsdaten {im} zurück?", "close"),
     ("Mein Kennwort für {akk} funktioniert nicht mehr.", "paraphrase")],
    [("{name} geht nicht", "colloquial"),
     ("Welche Priorität bekommt eine Störung {im}?", "paraphrase"),
     ("Wie melde ich eine Störung {im}?", "close"),
     ("Was kann ich selbst tun, bevor ich wegen {akk} die IT anrufe?", "paraphrase")],
    [("Was ist bei einem {name}-Ausfall zu tun?", "close"),
     ("{name} ist komplett down, womit arbeiten wir jetzt?", "colloquial"),
     ("Was muss nach einem {name}-Ausfall nachgetragen werden?", "paraphrase"),
     ("Notfallbetrieb {name}: wie läuft das?", "colloquial")],
    [("Gibt es eine Schulung für {akk}?", "close"),
     ("Wer sind die Key-User für {akk}?", "paraphrase"),
     ("Ich bin neu, wie lerne ich {name}?", "colloquial"),
     ("Wo finde ich Schulungsunterlagen für {akk}?", "close")],
    [("Wird gespeichert, wer {im} welche Daten ansieht?", "paraphrase"),
     ("Wie lange werden die {name}-Protokolle aufbewahrt?", "close"),
     ("Welche Datenschutzregeln gelten {im}?", "close"),
     ("kann die IT sehen, was ich {im} mache?", "colloquial")],
]

SYSTEMS = [
    dict(doc="KIS_Handbuch.pdf", title="Krankenhausinformationssystem (KIS) – Handbuch",
         name="KIS", akk="das KIS", im="im KIS", verb="wird",
         subject="Das Krankenhausinformationssystem (KIS)",
         purpose="die gesamte Patientendokumentation, von der Aufnahme über Anordnungen und "
                 "Pflegedokumentation bis zum Arztbrief",
         owner="das Medizincontrolling", team="Klinische Anwendungen",
         overview_extra="Alle anderen klinischen Systeme wie PACS, LIS und RIS sind über Schnittstellen "
                        "an das KIS angebunden.",
         group="GG_KIS_<Rolle>", form="Berechtigung Klinische Anwendungen",
         roles="KIS-Arzt, KIS-Pflege, KIS-Verwaltung und KIS-Lesezugriff",
         approver="die Pflegedirektion (Pflegerollen) bzw. die Chefärztin oder den Chefarzt der "
                  "Fachabteilung (ärztliche Rollen)",
         days="einem Werktag", recert="einmal jährlich",
         login="Die Anmeldung am KIS erfolgt per Single Sign-On mit dem Windows-Konto. Auf Station genügt "
               "es, den Mitarbeiterausweis an den Kartenleser zu halten, um schnell den Benutzer zu wechseln. "
               "Nach 15 Minuten ohne Aktivität wird die KIS-Sitzung aus Datenschutzgründen gesperrt.",
         reset="Das KIS nutzt das Windows-Passwort, ein eigenes KIS-Passwort gibt es nicht. Wer das "
               "Windows-Passwort vergessen hat, nutzt den Self-Service-Passwortreset im Intranet. Für die "
               "Signatur von Arztbriefen wird zusätzlich die Ausweis-PIN benötigt; eine gesperrte PIN "
               "entsperrt der Ausweisservice an der Pforte.",
         first_aid="das KIS vollständig schließen und neu starten, hilft das nicht, den PC neu starten",
         prio_condition="die Patientenversorgung auf einer ganzen Station behindern", prio="1",
         fallback="Fällt das KIS aus, arbeiten die Stationen mit den Papier-Notfallformularen aus dem "
                  "roten Notfallordner. Auf jeder Station steht ein Downtime-Rechner mit einem aktuellen "
                  "Lesestand der Patientendaten.",
         recovery="müssen alle auf Papier dokumentierten Daten innerhalb von 24 Stunden im KIS "
                  "nacherfasst werden.",
         training="besuchen vor der ersten Nutzung die zweistündige KIS-Grundschulung, die monatlich "
                  "stattfindet und über das Fortbildungsportal gebucht wird",
         keyuser="jeder Station", intranet="Anwendungen > KIS > Schulung", log="zehn Jahre",
         privacy="Der Zugriff auf Akten von Patientinnen und Patienten, die nicht auf der eigenen Station "
                 "liegen, ist nur mit Begründung über die Funktion „Notfallzugriff“ möglich und wird vom "
                 "Datenschutz stichprobenartig geprüft.",
         faq=[("Kann ich einen Arztbrief nach der Freigabe noch ändern?",
               "Nein. Nach der Freigabe ist der Arztbrief gesperrt; Korrekturen erfolgen über einen "
               "Nachtrag, den die Oberärztin oder der Oberarzt erneut signiert.",
               "Arztbrief schon freigegeben, aber ein Fehler drin – was nun?", "colloquial"),
              ("Warum sehe ich einen Patienten doppelt?",
               "Doppelt angelegte Patienten (Dubletten) meldet man der Patientenverwaltung; nur sie darf "
               "Dubletten zusammenführen.",
               "Patient ist zweimal im System angelegt, wer korrigiert das?", "paraphrase")]),

    dict(doc="PACS_Handbuch.pdf", title="Bildarchiv PACS – Handbuch",
         name="PACS", akk="das PACS", im="im PACS", verb="wird",
         subject="Das Bildarchivierungs- und Kommunikationssystem (PACS)",
         purpose="die Speicherung und Anzeige aller radiologischen Bilder im DICOM-Format, zum Beispiel "
                 "aus CT, MRT, Röntgen und Ultraschall",
         owner="die Klinik für Radiologie", team="Medizintechnik-IT",
         overview_extra="Bilder können über den Web-Viewer an jedem Klinik-PC oder an den "
                        "Befundungsarbeitsplätzen der Radiologie angesehen werden.",
         group="GG_PACS_<Rolle>", form="Berechtigung Medizinische Anwendungen",
         roles="PACS-Betrachter, PACS-Befunder und PACS-MTRA",
         approver="die Leitung der Radiologie", days="zwei Werktagen", recert="halbjährlich",
         login="Der Web-Viewer öffnet sich per Single Sign-On direkt aus dem KIS über die Schaltfläche "
               "„Bilder“. An den Befundungsarbeitsplätzen melden sich Radiologinnen und Radiologen "
               "zusätzlich mit ihrem persönlichen PACS-Kennwort an.",
         reset="Das persönliche PACS-Kennwort der Befundungsarbeitsplätze setzt die Medizintechnik-IT "
               "zurück; die Anfrage läuft über den IT-Servicedesk. Für den Web-Viewer gilt das "
               "Windows-Passwort.",
         first_aid="den Browser-Cache leeren und sich erneut anmelden",
         prio_condition="die Befundung verhindern", prio="1",
         fallback="Fällt das PACS aus, können aktuelle Aufnahmen direkt an der Konsole der Modalität "
                  "angesehen werden. Dringende Befunde werden telefonisch an die anfordernde Station "
                  "übermittelt.",
         recovery="sendet die Medizintechnik-IT alle zwischengespeicherten Aufnahmen der Modalitäten "
                  "automatisch an das PACS nach; die Radiologie prüft die Vollständigkeit.",
         training="erhalten eine 30-minütige Einweisung in den Web-Viewer durch die Radiologie; Befunder "
                  "werden an den Befundungsarbeitsplätzen gesondert eingewiesen",
         keyuser="der Radiologie", intranet="Anwendungen > PACS > Anleitungen", log="zehn Jahre",
         privacy="Bilddaten für externe Stellen werden nur über die Brennstation der Radiologie oder das "
                 "Teleradiologie-Portal weitergegeben. Für Forschungsprojekte müssen Bilder vorher über "
                 "die Pseudonymisierungsfunktion anonymisiert werden.",
         faq=[("Wie bekommt ein Patient seine Bilder auf CD?",
               "Die Brennstation in der Radiologie erstellt CDs oder DVDs; der Auftrag wird an der "
               "Anmeldung der Radiologie abgegeben.",
               "Patientin möchte ihre MRT-Bilder mit nach Hause nehmen", "paraphrase"),
              ("Darf ich Bilder auf einen USB-Stick exportieren?",
               "Nein, der Export auf USB-Sticks ist aus dem PACS nicht erlaubt.",
               "Kann ich CT-Bilder auf meinen Stick ziehen?", "colloquial")]),

    dict(doc="LIS_Labor_Handbuch.pdf", title="Laborinformationssystem (LIS) – Handbuch",
         name="LIS", akk="das LIS", im="im LIS", verb="wird",
         subject="Das Laborinformationssystem (LIS)",
         purpose="die Anforderung von Laboruntersuchungen, die Verwaltung der Proben und die "
                 "Übermittlung der Laborbefunde",
         owner="das Zentrallabor", team="Klinische Anwendungen",
         overview_extra="Laboranforderungen werden im KIS gestellt; die Befunde erscheinen nach der "
                        "technischen und medizinischen Validierung automatisch in der Patientenakte.",
         group="GG_LIS_<Rolle>", form="Berechtigung Labor",
         roles="LIS-Anforderer, LIS-MTLA und LIS-Validierer",
         approver="die Laborleitung", days="drei Werktagen", recert="einmal jährlich",
         login="Stationen nutzen das LIS nicht direkt, sondern über das KIS. Nur Laborpersonal meldet sich "
               "mit dem Windows-Konto am LIS-Client an; Validierer bestätigen jede Befundfreigabe "
               "zusätzlich mit ihrer Ausweis-PIN.",
         reset="Das LIS nutzt das Windows-Passwort; der Reset erfolgt über den Self-Service-Passwortreset "
               "im Intranet. Eine gesperrte Ausweis-PIN für die Befundfreigabe entsperrt der "
               "Ausweisservice an der Pforte.",
         first_aid="prüfen, ob der Etikettendrucker für Probenetiketten eingeschaltet ist und Papier hat, "
                   "und den LIS-Client neu starten",
         prio_condition="die Notfalldiagnostik betreffen", prio="1",
         fallback="Fällt das LIS aus, werden Laboranforderungen auf dem gelben Papier-Anforderungsschein "
                  "gestellt und die Proben mit handschriftlich beschrifteten Etiketten ins Labor gebracht. "
                  "Notfallwerte telefoniert das Labor unter der Durchwahl 5100 an die Station durch.",
         recovery="erfasst das Labor die Papieranforderungen nach und übermittelt die Befunde "
                  "nachträglich an das KIS.",
         training="im Labor werden durch die Laborleitung am Arbeitsplatz eingearbeitet; für "
                  "Stationspersonal gibt es eine kurze E-Learning-Einheit zur Laboranforderung im KIS",
         keyuser="des Zentrallabors", intranet="Anwendungen > Labor > Anleitungen", log="zehn Jahre",
         privacy="Befunde meldepflichtiger Erreger werden automatisch an das Hygieneteam weitergeleitet. "
                 "Genetische Befunde sind nur für die behandelnde Fachabteilung sichtbar.",
         faq=[("Warum sehe ich einen Laborwert noch nicht?",
               "Befunde erscheinen erst nach der medizinischen Validierung. Eilige Werte können "
               "telefonisch unter 5100 im Labor erfragt werden.",
               "Laborwerte fehlen noch in der Akte, obwohl die Probe längst weg ist", "paraphrase"),
              ("Wie storniere ich eine Laboranforderung?",
               "Solange die Probe noch nicht im Labor eingegangen ist, kann die Anforderung im KIS "
               "storniert werden; danach nur telefonisch über das Labor.",
               "Labor falsch angefordert, wie mache ich das rückgängig?", "colloquial")]),

    dict(doc="RIS_Radiologie_Handbuch.pdf", title="Radiologieinformationssystem (RIS) – Handbuch",
         name="RIS", akk="das RIS", im="im RIS", verb="wird",
         subject="Das Radiologieinformationssystem (RIS)",
         purpose="die Terminplanung, Leistungserfassung und Befundschreibung der Radiologie",
         owner="die Klinik für Radiologie", team="Medizintechnik-IT",
         overview_extra="Das RIS ist eng mit dem PACS verbunden: Das RIS verwaltet Aufträge und "
                        "Befundtexte, das PACS die Bilder.",
         group="GG_RIS_<Rolle>", form="Berechtigung Medizinische Anwendungen",
         roles="RIS-Terminplanung, RIS-MTRA und RIS-Befunder",
         approver="die Leitende MTRA (Planungs- und MTRA-Rollen) bzw. die Leitung der Radiologie "
                  "(Befunder)",
         days="zwei Werktagen", recert="halbjährlich",
         login="Die Anmeldung am RIS erfolgt mit dem Windows-Konto. Befundtexte werden per "
               "Spracherkennung diktiert; das Headset muss vor der Anmeldung angeschlossen sein. Nach 20 "
               "Minuten ohne Aktivität wird die RIS-Sitzung gesperrt.",
         reset="Das RIS nutzt das Windows-Passwort; der Reset erfolgt über den Self-Service-Passwortreset "
               "im Intranet. Probleme mit dem Spracherkennungsprofil behebt die Medizintechnik-IT.",
         first_aid="das Headset ab- und wieder anstecken und das RIS neu starten",
         prio_condition="die Notfallradiologie betreffen", prio="1",
         fallback="Fällt das RIS aus, werden Untersuchungen auf dem Papier-Auftragsbogen angenommen; "
                  "Befunde werden in Word diktiert und ausgedruckt der Station übergeben.",
         recovery="werden alle Untersuchungen und Befunde innerhalb von zwei Werktagen im RIS nacherfasst.",
         training="der Radiologie werden von der Leitenden MTRA eingearbeitet; für die Spracherkennung "
                  "gibt es ein 20-minütiges Training zum Anlegen des Sprachprofils",
         keyuser="der Radiologie", intranet="Anwendungen > RIS > Handbuch", log="zehn Jahre",
         privacy="Diktate der Spracherkennung werden nach der Freigabe des Befunds automatisch gelöscht.",
         faq=[("Wie vergebe ich einen Termin für ein MRT?",
               "MRT-Termine vergibt die Terminplanung der Radiologie im RIS; Stationen stellen die "
               "Anforderung im KIS.",
               "Station braucht einen MRT-Termin für einen Patienten, wer plant den?", "paraphrase"),
              ("Die Spracherkennung versteht mich schlecht, was tun?",
               "Das Sprachprofil kann über das Menü Extras > Profil trainieren verbessert werden.",
               "Diktat erkennt meine Wörter falsch", "colloquial")]),

    dict(doc="DMS_Archiv_Handbuch.pdf", title="Dokumentenmanagement und Archiv (DMS) – Handbuch",
         name="DMS", akk="das DMS", im="im DMS", verb="wird",
         subject="Das Dokumentenmanagementsystem (DMS)",
         purpose="die digitale Archivierung abgeschlossener Patientenakten und eingescannter Dokumente",
         owner="das Zentralarchiv", team="Klinische Anwendungen",
         overview_extra="Papierdokumente wie Einwilligungen und externe Befunde werden in der Scanstelle "
                        "eingescannt und der Fallnummer im DMS zugeordnet.",
         group="GG_DMS_<Rolle>", form="Berechtigung Archiv",
         roles="DMS-Lesen, DMS-Scannen und DMS-Archivverwaltung",
         approver="die Leitung des Zentralarchivs", days="drei Werktagen", recert="einmal jährlich",
         login="Das DMS wird aus dem KIS heraus über die Schaltfläche „Archiv“ aufgerufen; eine eigene "
               "Anmeldung ist nicht nötig. Die Scanstelle arbeitet mit einem separaten DMS-Client.",
         reset="Da das DMS über das KIS geöffnet wird, gilt das Windows-Passwort. Probleme mit dem "
               "Scan-Client meldet die Scanstelle dem IT-Servicedesk.",
         first_aid="das DMS-Fenster schließen und erneut aus dem KIS öffnen",
         prio_condition="den Zugriff auf Vorbefunde in der Notaufnahme verhindern", prio="2",
         fallback="Fällt das DMS aus, können archivierte Akten nicht eingesehen werden. Dringend benötigte "
                  "Unterlagen fordert man telefonisch beim Zentralarchiv an, das eine Notfallkopie "
                  "bereitstellt. Die Scanstelle sammelt die Papierdokumente.",
         recovery="scannt die Scanstelle alle gesammelten Dokumente innerhalb einer Woche nach.",
         training="erhalten eine kurze Einweisung durch die Key-User ihrer Abteilung; die Scanstelle "
                  "wird vom Zentralarchiv geschult",
         keyuser="des Zentralarchivs", intranet="Anwendungen > Archiv > Hilfe",
         log="dreißig Jahre, entsprechend der Aufbewahrungsfrist für Patientenakten",
         privacy="Die Löschung von Akten nach Ablauf der Aufbewahrungsfrist erfolgt automatisch nach "
                 "Freigabe durch den Datenschutz.",
         faq=[("Wie lange werden Patientenakten aufbewahrt?",
               "Patientenakten werden 30 Jahre nach Abschluss der Behandlung aufbewahrt.",
               "Wie viele Jahre bleiben alte Akten im Archiv?", "paraphrase"),
              ("Ein Dokument wurde dem falschen Patienten zugeordnet.",
               "Falsch zugeordnete Dokumente meldet man dem Zentralarchiv; nur die Archivverwaltung darf "
               "Dokumente umhängen.",
               "Befund im falschen Fall eingescannt, wer korrigiert das?", "colloquial")]),

    dict(doc="Dienstplan_Handbuch.pdf", title="Dienstplanprogramm und Mitarbeiterportal – Handbuch",
         name="Dienstplan", akk="das Dienstplanprogramm", im="im Dienstplanprogramm", verb="wird",
         subject="Das Dienstplanprogramm",
         purpose="die Planung von Schichten, Bereitschaftsdiensten und Abwesenheiten",
         owner="die Personalabteilung", team="Verwaltungsanwendungen",
         overview_extra="Mitarbeitende sehen ihren Dienstplan im Mitarbeiterportal auch von zu Hause und "
                        "auf dem Smartphone.",
         group="GG_DP_<Rolle>", form="Berechtigung Verwaltungsanwendungen",
         roles="Dienstplan-Ansicht, Dienstplan-Planer und Dienstplan-Administration",
         approver="die Personalabteilung", days="fünf Werktagen", recert="einmal jährlich",
         login="Planerinnen und Planer melden sich mit dem Windows-Konto am Dienstplanprogramm an. Alle "
               "anderen Mitarbeitenden nutzen das Mitarbeiterportal mit Benutzername und "
               "Portal-Passwort, auch außerhalb des Kliniknetzes.",
         reset="Das Portal-Passwort des Mitarbeiterportals ist unabhängig vom Windows-Passwort. Es wird "
               "über den Link „Passwort vergessen“ auf der Anmeldeseite per E-Mail an die private Adresse "
               "zurückgesetzt, die in der Personalabteilung hinterlegt ist.",
         first_aid="sich im Mitarbeiterportal ab- und wieder anmelden und einen anderen Browser versuchen",
         prio_condition="die Dienstplanung für den laufenden Tag verhindern", prio="2",
         fallback="Fällt das Dienstplanprogramm aus, gilt der zuletzt ausgedruckte Wochenplan, der in jeder "
                  "Abteilung aushängt. Tauschwünsche werden in dieser Zeit schriftlich bei der "
                  "Stationsleitung eingereicht.",
         recovery="tragen die Planerinnen und Planer alle Änderungen innerhalb von drei Werktagen nach.",
         training="mit Planungsaufgaben besuchen eine halbtägige Planerschulung der Personalabteilung; "
                  "für das Mitarbeiterportal gibt es eine Kurzanleitung",
         keyuser="der Personalabteilung", intranet="Personal > Dienstplan > Anleitungen",
         log="drei Jahre",
         privacy="Abwesenheitsgründe wie Krankheit sind nur für die Personalabteilung sichtbar; "
                 "Planerinnen und Planer sehen lediglich „abwesend“.",
         faq=[("Wie tausche ich einen Dienst?",
               "Diensttausche werden im Mitarbeiterportal beantragt; beide Beteiligten und die "
               "Stationsleitung müssen zustimmen.",
               "Will mit einer Kollegin die Schicht tauschen, geht das online?", "colloquial"),
              ("Wo sehe ich meine Überstunden?",
               "Das Arbeitszeitkonto mit Überstunden steht im Mitarbeiterportal unter „Mein Zeitkonto“.",
               "Wie viele Überstunden habe ich gerade?", "paraphrase")]),

    dict(doc="EMail_Outlook_Handbuch.pdf", title="E-Mail und Kalender (Outlook) – Handbuch",
         name="E-Mail", akk="Outlook", im="in Outlook", verb="wird",
         subject="Der E-Mail-Dienst mit Outlook",
         purpose="die dienstliche E-Mail-Kommunikation, Kalender und Besprechungsplanung",
         owner="die IT-Abteilung", team="Infrastruktur",
         overview_extra="Jede Mitarbeiterin und jeder Mitarbeiter erhält ein persönliches Postfach; "
                        "Abteilungen können zusätzlich Funktionspostfächer nutzen.",
         group="GG_MBX_<Postfach>", form="Funktionspostfach beantragen",
         roles="Lesen, Senden im Auftrag und Vollzugriff auf Funktionspostfächer; das persönliche "
               "Postfach erhält jede Person automatisch",
         approver="die oder den Verantwortlichen des Funktionspostfachs", days="einem Werktag",
         recert="einmal jährlich",
         login="Outlook meldet sich an Klinik-PCs automatisch mit dem Windows-Konto an. Auf dem "
               "Diensthandy wird die Outlook-App genutzt; die Anmeldung erfordert eine Bestätigung in der "
               "Authenticator-App. Outlook im Web ist auch von außerhalb erreichbar und meldet nach 60 "
               "Minuten ohne Aktivität automatisch ab.",
         reset="Outlook nutzt das Windows-Passwort; der Reset erfolgt über den Self-Service-Passwortreset "
               "im Intranet. Nach dem Reset muss das neue Passwort auch in der Outlook-App auf dem "
               "Diensthandy eingegeben werden.",
         first_aid="Outlook schließen, neu öffnen und prüfen, ob unten rechts „Verbunden“ angezeigt wird",
         prio_condition="den E-Mail-Versand im ganzen Haus verhindern", prio="2",
         fallback="Fällt der E-Mail-Dienst aus, werden dringende Informationen telefonisch oder über das "
                  "Intranet verteilt. E-Mails, die in dieser Zeit eingehen, werden zwischengespeichert "
                  "und gehen nicht verloren.",
         recovery="werden zwischengespeicherte E-Mails automatisch zugestellt; es ist nichts nachzutragen.",
         training="erhalten zum Start eine Kurzanleitung für Outlook; bei Bedarf bietet die IT eine "
                  "einstündige Online-Schulung an",
         keyuser="der IT-Abteilung", intranet="IT > E-Mail > Anleitungen", log="90 Tage",
         privacy="Patientendaten dürfen per E-Mail nur verschlüsselt versendet werden; dazu wird im Betreff "
                 "das Wort [vertraulich] vorangestellt. Die automatische Weiterleitung an private "
                 "Adressen ist technisch gesperrt.",
         faq=[("Wie richte ich eine Abwesenheitsnotiz ein?",
               "In Outlook unter Datei > Automatische Antworten.",
               "Bin im Urlaub, wie stelle ich eine automatische Antwort ein?", "paraphrase"),
              ("Mein Postfach ist voll.",
               "Das Postfach ist auf 10 GB begrenzt; alte E-Mails können über das Online-Archiv "
               "ausgelagert werden.",
               "Outlook sagt Postfach voll", "colloquial")]),

    dict(doc="VPN_Fernzugriff_Handbuch.pdf", title="VPN-Fernzugriff und Homeoffice – Handbuch",
         name="VPN", akk="das VPN", im="im VPN", verb="wird",
         subject="Der VPN-Fernzugriff",
         purpose="den sicheren Zugriff auf Kliniksysteme von außerhalb, etwa aus dem Homeoffice oder bei "
                 "Rufbereitschaft",
         owner="die IT-Abteilung", team="Netzwerk und Sicherheit",
         overview_extra="Der VPN-Zugang funktioniert ausschließlich mit dienstlichen Laptops; private "
                        "Geräte können sich nicht verbinden.",
         group="GG_VPN_Nutzer", form="Fernzugriff beantragen",
         roles="VPN-Standard (Büroanwendungen) und VPN-Klinisch (zusätzlich KIS und PACS)",
         approver="die oder den Vorgesetzten, für VPN-Klinisch zusätzlich durch die oder den "
                  "Informationssicherheitsbeauftragten",
         days="drei Werktagen", recert="halbjährlich",
         login="Die VPN-Verbindung wird über das Programm „Klinik-VPN“ gestartet. Nach Eingabe von "
               "Benutzername und Windows-Passwort muss die Anmeldung in der Authenticator-App bestätigt "
               "werden (Zwei-Faktor-Authentifizierung). Nach 8 Stunden wird die Verbindung automatisch "
               "getrennt.",
         reset="Für das VPN gilt das Windows-Passwort. Bei einem neuen Smartphone muss die "
               "Authenticator-App neu registriert werden; dafür ist ein Anruf beim IT-Servicedesk mit "
               "Identitätsprüfung nötig.",
         first_aid="die Internetverbindung zu Hause prüfen und das Programm Klinik-VPN neu starten",
         prio_condition="die Rufbereitschaft von Ärztinnen und Ärzten behindern", prio="2",
         fallback="Ist das VPN gestört, ist kein Fernzugriff möglich. Mitarbeitende in Rufbereitschaft "
                  "kommen in diesem Fall bei Bedarf ins Haus; die Telefonzentrale informiert über die "
                  "Dauer der Störung.",
         recovery="ist nichts nachzutragen; unterbrochene Sitzungen müssen neu gestartet werden.",
         training="mit VPN-Zugang erhalten bei der Ausgabe des Dienstlaptops eine Einweisung und eine "
                  "Anleitung zur Einrichtung der Authenticator-App",
         keyuser="der IT-Abteilung (Team Netzwerk und Sicherheit)",
         intranet="IT > Fernzugriff > Anleitungen", log="ein Jahr",
         privacy="Im Homeoffice dürfen Patientendaten nicht ausgedruckt werden, und der Bildschirm muss "
                 "vor Einsicht durch Dritte geschützt sein.",
         faq=[("Kann ich das VPN im Ausland nutzen?",
               "Nur nach vorheriger Genehmigung durch die oder den Informationssicherheitsbeauftragten; "
               "ohne Genehmigung sind Verbindungen aus dem Ausland gesperrt.",
               "Bin im Urlaub in Spanien, geht das VPN da?", "colloquial"),
              ("Warum ist mein VPN so langsam?",
               "Große Dateien sollten nicht über das VPN kopiert werden; Videokonferenzen laufen am besten "
               "direkt über das Internet ohne VPN.",
               "VPN extrem langsam bei Online-Meetings", "colloquial")]),

    dict(doc="Netzlaufwerke_Handbuch.pdf", title="Netzlaufwerke und Dateiablage – Handbuch",
         name="Netzlaufwerk", akk="die Netzlaufwerke", im="auf den Netzlaufwerken", verb="werden",
         subject="Die Netzlaufwerke",
         purpose="die zentrale Ablage von Dienstdokumenten wie Formularen, Protokollen und "
                 "Abteilungsunterlagen",
         owner="die IT-Abteilung", team="Infrastruktur",
         overview_extra="Jede Person hat ein persönliches Laufwerk H:, jede Abteilung ein "
                        "Abteilungslaufwerk G:. Für den Austausch mit externen Partnern steht die "
                        "klinikeigene Nextcloud zur Verfügung.",
         group="GG_FS_<Abteilung>", form="Ordnerberechtigung beantragen",
         roles="Lesen und Ändern",
         approver="die Ordnerverantwortliche oder den Ordnerverantwortlichen der Abteilung",
         days="einem Werktag", recert="einmal jährlich",
         login="Die Laufwerke werden bei der Windows-Anmeldung automatisch verbunden. Fehlt ein Laufwerk, "
               "hilft es, sich ab- und wieder anzumelden. Von zu Hause sind die Laufwerke nur über das "
               "VPN erreichbar.",
         reset="Für die Netzlaufwerke gilt das Windows-Passwort; der Reset erfolgt über den "
               "Self-Service-Passwortreset im Intranet. Wer trotz bestehender Berechtigung keinen Zugriff "
               "mehr auf einen Ordner hat, meldet sich beim IT-Servicedesk.",
         first_aid="im Explorer auf „Aktualisieren“ klicken und sich bei fehlendem Laufwerk ab- und wieder "
                   "anmelden",
         prio_condition="ein ganzes Abteilungslaufwerk unerreichbar machen", prio="2",
         fallback="Sind die Netzlaufwerke nicht erreichbar, können die wichtigsten Formulare aus dem "
                  "Intranet heruntergeladen werden. Dokumente werden vorübergehend lokal auf dem Desktop "
                  "gespeichert.",
         recovery="müssen lokal gespeicherte Dokumente auf das Netzlaufwerk verschoben und vom Desktop "
                  "gelöscht werden.",
         training="erhalten beim Start eine Kurzanleitung zur Ablagestruktur ihrer Abteilung von der oder "
                  "dem Ordnerverantwortlichen",
         keyuser="der jeweiligen Abteilung (Ordnerverantwortliche)",
         intranet="IT > Dateiablage > Anleitungen", log="ein Jahr",
         privacy="Patientendaten dürfen nicht auf dem persönlichen Laufwerk H: gespeichert werden, sondern "
                 "nur im KIS oder im DMS.",
         faq=[("Ich habe eine Datei gelöscht.",
               "Über Rechtsklick auf den Ordner > Vorgängerversionen lassen sich Dateien der letzten 30 "
               "Tage wiederherstellen.",
               "Datei aus Versehen gelöscht, kann man die zurückholen?", "colloquial"),
              ("Wie groß darf mein Laufwerk H: sein?",
               "Das persönliche Laufwerk H: ist auf 5 GB begrenzt; größere Datenmengen gehören auf das "
               "Abteilungslaufwerk.",
               "Mein H-Laufwerk ist voll", "colloquial")]),

    dict(doc="SAP_Materialwirtschaft_Handbuch.pdf", title="SAP Materialwirtschaft – Handbuch",
         name="SAP", akk="SAP", im="in SAP", verb="wird",
         subject="SAP Materialwirtschaft",
         purpose="die Bestellung von Verbrauchsmaterial, Medikamenten und Bürobedarf sowie die "
                 "Lagerverwaltung",
         owner="die Abteilung Einkauf und Logistik", team="Verwaltungsanwendungen",
         overview_extra="Stationen bestellen Material über den Stationswarenkorb; Medikamente werden über "
                        "die Apothekenanforderung in SAP bestellt.",
         group="GG_SAP_<Rolle>", form="Berechtigung SAP",
         roles="SAP-Anforderer, SAP-Freigeber und SAP-Einkauf",
         approver="die Kostenstellenverantwortliche oder den Kostenstellenverantwortlichen",
         days="fünf Werktagen", recert="einmal jährlich",
         login="Die Anmeldung erfolgt über das Programm SAP Logon per Single Sign-On mit dem Windows-Konto. "
               "Nach 60 Minuten ohne Aktivität wird die SAP-Sitzung beendet; nicht gespeicherte "
               "Bestellungen gehen dabei verloren.",
         reset="SAP nutzt Single Sign-On; ein eigenes Passwort gibt es nur für Notfallkonten des Einkaufs. "
               "Ist ein SAP-Benutzer gesperrt, entsperrt ihn das Team Verwaltungsanwendungen über ein "
               "Ticket beim IT-Servicedesk.",
         first_aid="das SAP-Fenster schließen, SAP Logon neu starten und die Transaktion erneut aufrufen",
         prio_condition="die Versorgung mit Medikamenten oder Sterilgut gefährden", prio="1",
         fallback="Fällt SAP aus, werden dringende Bestellungen telefonisch beim Zentrallager (Durchwahl "
                  "6200) oder bei der Apotheke aufgegeben und auf dem Notfall-Bestellschein dokumentiert.",
         recovery="erfasst das Zentrallager die telefonischen Bestellungen innerhalb von zwei Werktagen "
                  "nachträglich in SAP.",
         training="mit Bestellaufgaben nehmen an einer zweistündigen SAP-Schulung des Einkaufs teil",
         keyuser="der Abteilung Einkauf und Logistik", intranet="Einkauf > SAP > Anleitungen",
         log="zehn Jahre, entsprechend den handelsrechtlichen Aufbewahrungsfristen",
         privacy="Bestellungen über 5.000 Euro benötigen eine zusätzliche Freigabe durch den Einkauf "
                 "(Vier-Augen-Prinzip).",
         faq=[("Wie verfolge ich den Status meiner Bestellung?",
               "Der Status ist im Stationswarenkorb unter „Meine Bestellungen“ sichtbar.",
               "Wo sehe ich, ob unsere Materialbestellung schon unterwegs ist?", "paraphrase"),
              ("Wie bestelle ich Medikamente?",
               "Medikamente werden in SAP über die Apothekenanforderung bestellt, nicht über den normalen "
               "Warenkorb.",
               "Station braucht Nachschub an Arzneimitteln, welcher Weg?", "paraphrase")]),
]

# =====================================================================================
# PART B - Two cross-cutting handbooks written by hand (no templates).
# Each section: (title, text, test question, test type, train question, train type)
# =====================================================================================

GENERAL_DOCS = [
    dict(doc="Servicedesk_Handbuch.pdf", title="IT-Servicedesk – Prozesse und Services", sections=[
        ("Kontakt und Servicezeiten",
         "Der IT-Servicedesk ist unter der Durchwahl 4444 erreichbar. Standardanfragen werden montags bis "
         "freitags von 7 bis 18 Uhr bearbeitet, Störungen der Prioritäten 1 und 2 rund um die Uhr. "
         "Schriftliche Anfragen gehen an servicedesk@klinikum-musterstadt.de oder über das "
         "Self-Service-Portal.",
         "IT Hotline Nummer?", "colloquial",
         "Wann ist der Servicedesk für normale Anfragen besetzt?", "close"),
        ("Self-Service-Portal",
         "Im Self-Service-Portal (Intranet > IT-Service) können Mitarbeitende Tickets erstellen, den Status "
         "eigener Tickets verfolgen, Hardware bestellen und Berechtigungen beantragen. Anträge für "
         "Berechtigungen dürfen nur Vorgesetzte stellen; alle anderen Funktionen stehen allen "
         "Mitarbeitenden offen.",
         "Wo sehe ich, was aus meinem Ticket geworden ist?", "paraphrase",
         "Was kann man alles im Self-Service-Portal machen?", "close"),
        ("Prioritäten und Reaktionszeiten",
         "Priorität 1 (kritisch): Die Patientenversorgung ist direkt beeinträchtigt; Reaktion innerhalb von "
         "30 Minuten, Lösung angestrebt in 4 Stunden. Priorität 2 (hoch): wichtige Funktionen fehlen für "
         "viele Nutzer; Reaktion in 2 Stunden, Lösung innerhalb eines Werktags. Priorität 3 (normal): "
         "Reaktion innerhalb eines Werktags, Lösung in 5 Werktagen. Priorität 4 (niedrig): Wünsche und "
         "Anfragen ohne Zeitdruck.",
         "Wie schnell muss die IT bei einem kritischen Problem reagieren?", "paraphrase",
         "Welche Ticket-Prioritäten gibt es?", "close"),
        ("Ein gutes Ticket schreiben",
         "Ein Ticket kann schneller gelöst werden, wenn es folgende Angaben enthält: Standort und Station, "
         "die Inventarnummer des Geräts vom Aufkleber auf dem Gerät, die genaue Fehlermeldung am besten als "
         "Screenshot, ob die Patientenversorgung betroffen ist, und eine Rückrufnummer.",
         "Was muss ich angeben, wenn ich der IT einen Fehler melde?", "paraphrase",
         "Wo steht die Inventarnummer von meinem PC?", "colloquial"),
        ("Eskalation und Rufbereitschaft",
         "Wird eine Reaktionszeit überschritten, kann das Ticket bei der Leitung des Servicedesks eskaliert "
         "werden. Außerhalb der Servicezeiten erreicht man die IT-Rufbereitschaft über die Telefonzentrale "
         "(Durchwahl 0).",
         "Nachts funktioniert nichts mehr, wen rufe ich an?", "colloquial",
         "An wen wende ich mich, wenn mein Ticket liegen bleibt?", "paraphrase"),
        ("Major Incident",
         "Betrifft eine Störung mehrere Bereiche gleichzeitig, ruft die IT einen Major Incident aus. Alle "
         "Stationen werden über ein Banner im Intranet und per Rundruf informiert; einzelne Tickets zur "
         "selben Störung sind dann nicht nötig. Nach der Behebung erstellt die IT einen Abschlussbericht.",
         "Muss jeder einzeln anrufen, wenn das ganze Haus betroffen ist?", "paraphrase",
         "Was ist ein Major Incident?", "close"),
        ("Neue Mitarbeitende",
         "Das Benutzerkonto wird automatisch am Werktag vor dem Eintritt angelegt, sobald die "
         "Personalabteilung den Vertrag erfasst hat. Der Benutzername besteht aus dem Nachnamen und dem "
         "ersten Buchstaben des Vornamens. Das Startpasswort übergibt die oder der Vorgesetzte persönlich. "
         "Anwendungsrechte wie KIS oder PACS müssen separat beantragt werden.",
         "Neuer Kollege fängt Montag an, hat der schon einen Account?", "colloquial",
         "Wie ist mein Benutzername aufgebaut?", "close"),
        ("Austritt und befristete Konten",
         "Beim Austritt wird das Benutzerkonto am letzten Arbeitstag um 18 Uhr deaktiviert, und alle "
         "Anwendungsrechte werden entfernt. Das Postfach bleibt 30 Tage erhalten. Gastärztinnen, Gastärzte "
         "und Hospitierende erhalten befristete Konten mit höchstens 90 Tagen Laufzeit.",
         "Wie lange gilt ein Konto für Gastärzte?", "close",
         "Was passiert mit meinem Account, wenn ich kündige?", "colloquial"),
        ("Hardware bestellen",
         "Hardware wie Monitore, Tastaturen oder Headsets wird im Warenkorb des Self-Service-Portals "
         "bestellt. Die Bestellung gibt die oder der Kostenstellenverantwortliche frei; die Lieferzeit "
         "beträgt etwa zehn Werktage.",
         "Ich hätte gern einen zweiten Bildschirm", "colloquial",
         "Wie lange dauert die Lieferung neuer Hardware?", "close"),
        ("Software installieren",
         "Mitarbeitende haben keine Administratorrechte. Freigegebene Programme installiert man selbst über "
         "das Softwarecenter. Für neue Software ist ein Antrag im Self-Service-Portal nötig; IT-Sicherheit "
         "und Datenschutz prüfen sie vor der Freigabe.",
         "Wie installiere ich ein Programm auf meinem PC?", "close",
         "Warum darf ich nichts installieren?", "colloquial"),
    ]),
    dict(doc="IT_Sicherheitsrichtlinie.pdf", title="IT-Sicherheitsrichtlinie des Klinikums", sections=[
        ("Geltungsbereich und Ansprechpartner",
         "Diese Richtlinie gilt für alle Mitarbeitenden, Auszubildenden, Gastärztinnen und Gastärzte sowie "
         "externe Dienstleister. Ansprechpartner für Fragen der Informationssicherheit ist die oder der "
         "Informationssicherheitsbeauftragte (ISB), erreichbar unter isb@klinikum-musterstadt.de. Verstöße "
         "können arbeitsrechtliche Folgen haben.",
         "Wer ist bei uns für IT-Sicherheit zuständig?", "paraphrase",
         "Gilt die Sicherheitsrichtlinie auch für externe Firmen?", "close"),
        ("Passwortregeln",
         "Passwörter müssen mindestens zwölf Zeichen lang sein und werden alle 180 Tage geändert. Sie "
         "dürfen nicht weitergegeben oder notiert werden. Vergessene Windows-Passwörter setzt man über den "
         "Self-Service-Passwortreset im Intranet zurück; dafür muss eine Mobilnummer hinterlegt sein.",
         "Wie lang muss mein Passwort sein?", "close",
         "Warum muss ich mein Kennwort ständig ändern?", "colloquial"),
        ("Umgang mit Patientendaten",
         "Patientendaten unterliegen der ärztlichen Schweigepflicht und der DSGVO. Sie dürfen nicht über "
         "private E-Mail-Konten oder Messenger wie WhatsApp versendet werden und nur so lange gespeichert "
         "werden, wie es für die Behandlung nötig ist.",
         "Darf ich ein Foto einer Wunde per WhatsApp an den Oberarzt schicken?", "paraphrase",
         "Welche Regeln gelten für Patientendaten?", "close"),
        ("Phishing und verdächtige E-Mails",
         "Verdächtige E-Mails werden über die Schaltfläche „Phishing melden“ in Outlook weitergeleitet. "
         "Links und Anhänge solcher E-Mails dürfen nicht geöffnet werden. Wurde bereits geklickt, sofort "
         "den IT-Servicedesk unter 4444 anrufen, damit das Gerät isoliert werden kann.",
         "Komische Mail mit Link bekommen, was mache ich?", "colloquial",
         "Wie melde ich eine Phishing-Mail?", "close"),
        ("USB-Datenträger",
         "USB-Anschlüsse sind für Speichermedien gesperrt. Ausnahmen genehmigt die oder der "
         "Informationssicherheitsbeauftragte; dann dürfen nur verschlüsselte, vom Klinikum ausgegebene "
         "USB-Sticks verwendet werden.",
         "Mein USB-Stick wird am PC nicht erkannt", "colloquial",
         "Wer genehmigt die Nutzung von USB-Sticks?", "paraphrase"),
        ("Cloud-Dienste und Dateiaustausch",
         "Private Cloud-Dienste wie Dropbox, Google Drive oder WeTransfer sind für dienstliche Daten "
         "verboten. Für den Dateiaustausch mit externen Partnern steht die klinikeigene Nextcloud zur "
         "Verfügung; Freigabelinks laufen nach 14 Tagen ab.",
         "Wie schicke ich einer externen Firma eine große Datei?", "paraphrase",
         "Darf ich Dropbox benutzen?", "close"),
        ("Private Geräte und WLAN",
         "Private Smartphones, Tablets und Laptops dürfen nicht mit dem Kliniknetz verbunden werden. Das "
         "Gäste-WLAN „Klinikum-Gast“ darf privat genutzt werden, bietet aber keinen Zugriff auf "
         "Kliniksysteme.",
         "Kann ich mein privates Handy ins Klinik-WLAN bringen?", "colloquial",
         "Gibt es WLAN für private Geräte?", "close"),
        ("Arbeitsplatz und Ausdrucke",
         "Beim Verlassen des Arbeitsplatzes wird der Bildschirm mit Windows + L gesperrt. Ausdrucke mit "
         "Patientendaten dürfen nicht offen liegen bleiben und werden in den verschlossenen "
         "Datenschutztonnen entsorgt.",
         "Wohin mit alten Ausdrucken, auf denen Patientennamen stehen?", "paraphrase",
         "Wie sperre ich schnell meinen Bildschirm?", "close"),
        ("Sicherheitsvorfälle melden",
         "Jeder Verdacht auf einen Sicherheitsvorfall, etwa ein verlorenes Diensthandy, eine Virenmeldung "
         "oder ein unbekanntes Gerät im Netzwerk, ist sofort dem IT-Servicedesk unter 4444 zu melden, "
         "außerhalb der Servicezeiten über die Telefonzentrale. Verlorene Geräte werden aus der Ferne "
         "gesperrt und gelöscht.",
         "Diensthandy verloren, was jetzt?", "colloquial",
         "Was zählt als Sicherheitsvorfall?", "close"),
        ("Pflichtschulung",
         "Alle Mitarbeitenden absolvieren jährlich die E-Learning-Pflichtschulung Informationssicherheit "
         "und Datenschutz im Fortbildungsportal. Die Teilnahme wird der oder dem Vorgesetzten gemeldet; "
         "neue Mitarbeitende absolvieren sie in den ersten vier Wochen.",
         "Muss ich diese Datenschutz-Schulung wirklich jedes Jahr machen?", "colloquial",
         "Bis wann müssen neue Mitarbeitende die Sicherheitsschulung machen?", "paraphrase"),
    ]),
]

# Extra relevant pages for some handwritten test questions: (doc, section title) -> grade
EXTRA_QRELS = {
    "Wie schicke ich einer externen Firma eine große Datei?":
        {("Netzlaufwerke_Handbuch.pdf", "Überblick"): 1},
}

# =====================================================================================
# PART C - Harder test questions: several relevant pages (graded), or no answer at all.
# Grade 2 = contains the answer, grade 1 = useful context.
# =====================================================================================

MULTI_PAGE = [
    ("Neue Kollegin auf Station braucht KIS und PACS – wie bekommt sie beides?",
     {("KIS_Handbuch.pdf", "Zugriff beantragen"): 2, ("PACS_Handbuch.pdf", "Zugriff beantragen"): 2,
      ("Servicedesk_Handbuch.pdf", "Neue Mitarbeitende"): 1}),
    ("Wer genehmigt Zugriffe auf die Radiologie-Systeme?",
     {("PACS_Handbuch.pdf", "Zugriff beantragen"): 2, ("RIS_Radiologie_Handbuch.pdf", "Zugriff beantragen"): 2}),
    ("Ab nächster Woche Homeoffice – wie komme ich von zu Hause an meine Dateien?",
     {("VPN_Fernzugriff_Handbuch.pdf", "Zugriff beantragen"): 2, ("Netzlaufwerke_Handbuch.pdf", "Anmeldung"): 2}),
    ("Habe auf einen Link in einer verdächtigen Mail geklickt",
     {("IT_Sicherheitsrichtlinie.pdf", "Phishing und verdächtige E-Mails"): 2,
      ("IT_Sicherheitsrichtlinie.pdf", "Sicherheitsvorfälle melden"): 1}),
    ("Weder Laborwerte noch Bilder sind abrufbar – wie arbeiten wir weiter?",
     {("LIS_Labor_Handbuch.pdf", "Notfallbetrieb bei Ausfall"): 2,
      ("PACS_Handbuch.pdf", "Notfallbetrieb bei Ausfall"): 2}),
    ("Mein Windows-Passwort ist abgelaufen, jetzt komme ich auch nicht mehr ins KIS",
     {("IT_Sicherheitsrichtlinie.pdf", "Passwortregeln"): 2, ("KIS_Handbuch.pdf", "Zugangsdaten zurücksetzen"): 2}),
    ("Wie lange werden Zugriffe auf Patientenakten protokolliert?",
     {("KIS_Handbuch.pdf", "Datenschutz und Protokollierung"): 2,
      ("DMS_Archiv_Handbuch.pdf", "Datenschutz und Protokollierung"): 1}),
    ("KIS auf der ganzen Station ausgefallen – welche Priorität und wie schnell reagiert die IT?",
     {("KIS_Handbuch.pdf", "Störungen melden"): 2, ("Servicedesk_Handbuch.pdf", "Prioritäten und Reaktionszeiten"): 2}),
    ("Ein Oberarzt ist ausgeschieden – werden seine Zugänge automatisch gesperrt?",
     {("Servicedesk_Handbuch.pdf", "Austritt und befristete Konten"): 2,
      ("KIS_Handbuch.pdf", "Berechtigungen ändern und entziehen"): 1}),
    ("Wie schicke ich einer Partnerklinik Röntgenbilder, ohne gegen den Datenschutz zu verstoßen?",
     {("PACS_Handbuch.pdf", "Datenschutz und Protokollierung"): 2,
      ("IT_Sicherheitsrichtlinie.pdf", "Cloud-Dienste und Dateiaustausch"): 1}),
]

UNANSWERABLE = [
    "Wann ist die nächste Weihnachtsfeier?",
    "Wie hoch ist das Gehalt einer Pflegekraft?",
    "Wo ist die Cafeteria?",
    "Wie beantrage ich Elternzeit?",
    "Welche Parkplätze darf ich als Mitarbeiter nutzen?",
    "Wie melde ich mich für den Firmenlauf an?",
    "Wer ist Chefarzt der Kardiologie?",
    "Wie beantrage ich einen Dienstwagen?",
    # The hard ones: IT questions that sound answerable, but the handbooks don't cover them.
    "Welches Antivirenprogramm läuft auf den Servern?",
    "Wann wird Windows 12 im Klinikum eingeführt?",
    "Wie drucke ich in Farbe?",
    "Wie verbinde ich mein Diensthandy mit dem Auto per Bluetooth?",
]

# =====================================================================================
# Build everything
# =====================================================================================


def page_id(doc, page):
    return f"{doc}#{page}"


def build_pages():
    pages = []                                  # list of dicts, one per content page
    for s in SYSTEMS:
        for a, title in enumerate(ASPECTS):
            if PAGE_TEMPLATES[a] is None:       # FAQ page
                text = " ".join(f"Frage: {q} Antwort: {ans}" for q, ans, _, _ in s["faq"])
            else:
                text = PAGE_TEMPLATES[a].format(**s)
            pages.append(dict(doc=s["doc"], doc_title=s["title"], page=FIRST_PAGE + a,
                              title=title, text=text))
    for g in GENERAL_DOCS:
        for i, sec in enumerate(g["sections"]):
            pages.append(dict(doc=g["doc"], doc_title=g["title"], page=FIRST_PAGE + i,
                              title=sec[0], text=sec[1]))
    for p in pages:
        p["id"] = page_id(p["doc"], p["page"])
    return pages


def build_questions(pages):
    by_title = {(p["doc"], p["title"]): p["id"] for p in pages}
    questions = []

    def add(text, qtype, split, qrels):
        questions.append(dict(id=f"q{len(questions) + 1:04d}", question=text, type=qtype,
                              split=split, qrels=qrels))

    # Templated questions: one test (template 0 or 1) + one train (template 2 or 3) per page.
    for i, s in enumerate(SYSTEMS):
        for a, templates in enumerate(QUESTION_TEMPLATES):
            pid = by_title[(s["doc"], ASPECTS[a])]
            for split, t in (("test", i % 2), ("train", 2 + i % 2)):
                text, qtype = templates[t]
                add(text.format(**s), qtype, split, {pid: 2})
        faq_pid = by_title[(s["doc"], "Häufige Fragen")]
        for (_, _, q, qtype), split in zip(s["faq"], ("test", "train")):
            add(q, qtype, split, {faq_pid: 2})

    # Handwritten questions for the two general handbooks.
    for g in GENERAL_DOCS:
        for title, _, test_q, test_t, train_q, train_t in g["sections"]:
            pid = by_title[(g["doc"], title)]
            extra = {by_title[k]: v for k, v in EXTRA_QRELS.get(test_q, {}).items()}
            add(test_q, test_t, "test", {pid: 2} | extra)
            add(train_q, train_t, "train", {pid: 2})

    for text, rel in MULTI_PAGE:
        add(text, "multi_page", "test", {by_title[k]: v for k, v in rel.items()})
    for text in UNANSWERABLE:
        add(text, "unanswerable", "test", {})
    return questions


def write_pdf(path, doc_title, doc_pages):
    styles = getSampleStyleSheet()
    story = [Spacer(1, 200), Paragraph(doc_title, styles["Title"]),
             Paragraph("Klinikum Musterstadt · Abteilung IT · Version 2026.1 (fiktives Beispiel)",
                       styles["Normal"]),
             PageBreak(), Paragraph("Inhalt", styles["Heading1"])]
    for n, p in enumerate(doc_pages, start=1):
        story.append(Paragraph(f"{n}. {p['title']} ........ Seite {p['page']}", styles["Normal"]))
    for n, p in enumerate(doc_pages, start=1):
        story += [PageBreak(), Paragraph(f"{n}. {p['title']}", styles["Heading1"])]
        if p["title"] == "Häufige Fragen":           # one paragraph per Q/A for readability
            for chunk in p["text"].split("Frage: ")[1:]:
                story.append(Paragraph(escape("Frage: " + chunk.strip()), styles["BodyText"]))
        else:
            story.append(Paragraph(escape(p["text"]), styles["BodyText"]))  # escape: "<Rolle>" is not markup
    SimpleDocTemplate(str(path), pagesize=A4, title=doc_title).build(story)


def write_jsonl(path, rows, keys):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n")


def main():
    pages = build_pages()
    questions = build_questions(pages)

    # Sanity checks: every labelled page must exist, and no question text may appear twice.
    ids = {p["id"] for p in pages}
    assert all(pid in ids for q in questions for pid in q["qrels"]), "qrels point to a missing page"
    texts = [q["question"] for q in questions]
    assert len(texts) == len(set(texts)), "duplicate question text"

    (OUT / "docs").mkdir(exist_ok=True)
    for doc in dict.fromkeys(p["doc"] for p in pages):          # keeps document order
        doc_pages = [p for p in pages if p["doc"] == doc]
        write_pdf(OUT / "docs" / doc, doc_pages[0]["doc_title"], doc_pages)
    write_jsonl(OUT / "pages.jsonl", pages, ["id", "doc", "page", "title", "text"])
    write_jsonl(OUT / "questions.jsonl", questions, ["id", "question", "type", "split", "qrels"])

    # Summary
    n_docs = len({p["doc"] for p in pages})
    print(f"{n_docs} PDFs, {len(pages)} content pages -> random Recall@10 ≈ {10 / len(pages):.1%}")
    for split in ("test", "train"):
        qs = [q for q in questions if q["split"] == split]
        counts = {}
        for q in qs:
            counts[q["type"]] = counts.get(q["type"], 0) + 1
        print(f"{split:5s}: {len(qs):3d} questions  {counts}")


if __name__ == "__main__":
    main()
