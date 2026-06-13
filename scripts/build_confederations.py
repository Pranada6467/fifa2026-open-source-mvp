"""Generate data/raw/confederations.csv — static team→confederation map.

Every team that appears in data/processed/matches.parquet must resolve.
Non-FIFA entities (ConIFA, sub-national, historical) are assigned to the
geographic confederation.

Run:  .venv/bin/python scripts/build_confederations.py
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# FIFA confederations:
# AFC  — Asian Football Confederation
# CAF  — Confederation of African Football
# CONCACAF — North, Central America and Caribbean
# CONMEBOL — South American Football Confederation
# OFC  — Oceania Football Confederation
# UEFA — Union of European Football Associations

AFC = {
    "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei",
    "Cambodia", "China", "Guam", "Hong Kong", "India", "Indonesia", "Iran",
    "Iraq", "Japan", "Jordan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon",
    "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal",
    "North Korea", "Northern Mariana Islands", "Oman", "Pakistan",
    "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore",
    "South Korea", "Sri Lanka", "Syria", "Tajikistan", "Taiwan", "Thailand",
    "Timor-Leste", "Turkmenistan", "United Arab Emirates", "Uzbekistan",
    "Vietnam", "Yemen",
    # Historical / non-FIFA — geographic
    "Chagos Islands", "East Turkestan", "Hmong", "Iraqi Kurdistan",
    "Kurdistan", "Manchukuo", "North Vietnam", "Panjab", "Ryūkyū",
    "South Yemen", "Tamil Eelam", "Tibet", "United Koreans in Japan",
    "Vietnam Republic", "West Papua", "Western Australia", "Yemen DPR",
}

CAF = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
    "Congo", "DR Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea",
    "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
    "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya",
    "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco",
    "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
    "São Tomé and Príncipe", "Senegal", "Seychelles", "Sierra Leone",
    "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo",
    "Tunisia", "Uganda", "Zambia", "Zimbabwe",
    # Non-FIFA — geographic
    "Ambazonia", "Barawa", "Biafra", "Darfur", "Kabylia", "Matabeleland",
    "Mayotte", "Réunion", "Saint Helena", "Somaliland", "Western Sahara",
    "Yoruba Nation", "Zanzibar",
}

CONCACAF = {
    "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Belize",
    "Bermuda", "Bonaire", "British Virgin Islands", "Canada",
    "Cayman Islands", "Costa Rica", "Cuba", "Curaçao", "Dominica",
    "Dominican Republic", "El Salvador", "French Guiana", "Grenada",
    "Guadeloupe", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica",
    "Martinique", "Mexico", "Montserrat", "Nicaragua", "Panama",
    "Puerto Rico", "Saint Kitts and Nevis", "Saint Lucia", "Saint Martin",
    "Saint Vincent and the Grenadines", "Sint Maarten", "Suriname",
    "Trinidad and Tobago", "Turks and Caicos Islands", "United States",
    "United States Virgin Islands",
    # Non-FIFA — geographic
    "Anguilla", "Cascadia", "Greenland", "Quebec",
    "Saint Barthélemy", "Saint Pierre and Miquelon",
}

CONMEBOL = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
    "Paraguay", "Peru", "Uruguay", "Venezuela",
    # Non-FIFA — geographic
    "Aymara", "Falkland Islands", "Mapuche", "Maule Sur",
}

OFC = {
    "American Samoa", "Cook Islands", "Fiji", "Kiribati",
    "Marshall Islands", "Micronesia", "New Caledonia", "New Zealand", "Niue",
    "Palau", "Papua New Guinea", "Samoa", "Solomon Islands", "Tahiti",
    "Tonga", "Tuvalu", "Vanuatu",
    # Non-FIFA — geographic
    "Wallis Islands and Futuna",
}

UEFA = {
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
    "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
    "Czech Republic", "Denmark", "England", "Estonia", "Faroe Islands",
    "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece",
    "Hungary", "Iceland", "Israel", "Italy", "Kazakhstan", "Kosovo",
    "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova",
    "Monaco", "Montenegro", "Netherlands", "North Macedonia",
    "Northern Ireland", "Norway", "Poland", "Portugal",
    "Republic of Ireland", "Romania", "Russia", "San Marino", "Scotland",
    "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
    "Turkey", "Ukraine", "Wales",
    # Historical
    "Czechoslovakia", "German DR", "Yugoslavia",
    # Non-FIFA — geographic
    "Abkhazia", "Alderney", "Andalusia", "Arameans Suryoye", "Artsakh",
    "Asturias", "Basque Country", "Brittany", "Canary Islands", "Catalonia",
    "Central Spain", "Chameria", "Chechnya", "Cilento", "Corsica",
    "County of Nice", "Crimea", "Délvidék", "Donetsk PR", "Elba Island",
    "Ellan Vannin", "Felvidék", "Franconia", "Frøya", "Galicia",
    "Găgăuzia", "Gotland", "Gozo", "Guernsey", "Hitra", "Isle of Man",
    "Isle of Wight", "Jersey", "Kernow", "Kárpátalja", "Luhansk PR",
    "Madrid", "Menorca", "Northern Cyprus", "Occitania", "Orkney",
    "Padania", "Parishes of Jersey", "Provence", "Raetia",
    "Republic of St. Pauli", "Rhodes", "Romani people", "Saare County",
    "Saarland", "Sark", "Saugeais", "Sealand", "Seborga", "Shetland",
    "Silesia", "South Ossetia", "Surrey", "Székely Land", "Sápmi",
    "Ticino", "Two Sicilies", "Vatican City", "Western Armenia",
    "Western Isles", "Ynys Môn", "Yorkshire", "Åland Islands",
}

CONFED_MAP: dict[str, str] = {}
for confed, teams in [
    ("AFC", AFC), ("CAF", CAF), ("CONCACAF", CONCACAF),
    ("CONMEBOL", CONMEBOL), ("OFC", OFC), ("UEFA", UEFA),
]:
    for team in teams:
        if team in CONFED_MAP:
            raise ValueError(f"duplicate: {team} in {confed} and {CONFED_MAP[team]}")
        CONFED_MAP[team] = confed


def main():
    matches = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "matches.parquet")
    all_teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))

    missing = [t for t in all_teams if t not in CONFED_MAP]
    if missing:
        print(f"UNMAPPED teams ({len(missing)}):")
        for t in missing:
            print(f"  {t}")
        raise SystemExit(1)

    extra = set(CONFED_MAP) - set(all_teams)
    if extra:
        print(f"Warning: {len(extra)} mapped teams not in data: {sorted(extra)}")

    df = pd.DataFrame(
        [(t, CONFED_MAP[t]) for t in all_teams],
        columns=["team", "confederation"],
    )
    out = PROJECT_ROOT / "data" / "raw" / "confederations.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} teams to {out}")
    print(f"Confederation counts: {df['confederation'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
