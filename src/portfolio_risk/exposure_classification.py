"""Reconcile provider taxonomies for descriptive exposure, not investment factors."""

from __future__ import annotations

SECTORS = {
    "technology": "technology",
    "information technology": "technology",
    "financial services": "financials",
    "financials": "financials",
    "healthcare": "health_care",
    "health care": "health_care",
    "health_care": "health_care",
    "consumer cyclical": "consumer_discretionary",
    "consumer discretionary": "consumer_discretionary",
    "consumer_discretionary": "consumer_discretionary",
    "consumer defensive": "consumer_staples",
    "consumer staples": "consumer_staples",
    "consumer_staples": "consumer_staples",
    "communication services": "communication_services",
    "communication_services": "communication_services",
    "basic materials": "materials",
    "materials": "materials",
    "real estate": "real_estate",
    "real_estate": "real_estate",
    "industrial": "industrials",
    "industrials": "industrials",
    "energy": "energy",
    "utilities": "utilities",
}
COUNTRIES = {
    "united states": "US",
    "united states of america": "US",
    "us": "US",
    "canada": "CA",
    "ca": "CA",
    "united kingdom": "GB",
    "gb": "GB",
    "netherlands": "NL",
    "nl": "NL",
    "germany": "DE",
    "de": "DE",
    "france": "FR",
    "fr": "FR",
    "japan": "JP",
    "jp": "JP",
    "china": "CN",
    "cn": "CN",
    "taiwan": "TW",
    "tw": "TW",
    "australia": "AU",
    "au": "AU",
    "switzerland": "CH",
    "ch": "CH",
    "denmark": "DK",
    "dk": "DK",
    "india": "IN",
    "in": "IN",
    "ireland": "IE",
    "ie": "IE",
    "brazil": "BR",
    "br": "BR",
    "south korea": "KR",
    "korea, republic of": "KR",
    "kr": "KR",
}


def canonical_sector(value: str) -> str:
    return SECTORS.get(value.strip().lower(), value.strip().lower().replace(" ", "_"))


def canonical_country(value: str) -> str:
    return COUNTRIES.get(value.strip().lower(), value.strip())
