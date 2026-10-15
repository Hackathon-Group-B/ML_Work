"""A-02: extraction handles references, amounts, durations and bank names, including
Pidgin and typo variants. The PRD asks for at least 40 unit cases; this file has well over.
"""
import pytest

from ai.complaints.extract import (extract, extract_amount, extract_elapsed_minutes,
                                   extract_institution, extract_reference, mask_pii)
from ai.complaints.normalise import normalise, tag_language
from ml.config import load_config

CFG = load_config().complaints


# --- references ---------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("My ref is ZT839102", "ZT839102"),
    ("ref zt839102 please check", "ZT839102"),            # lower case
    ("ZT 839102", "ZT839102"),                            # space
    ("ZT-839102", "ZT839102"),                            # hyphen
    ("zt - 839102", "ZT839102"),                          # spaces around hyphen
    ("abeg check ZT839102 for me", "ZT839102"),           # Pidgin
    ("Ref:ZT839102.", "ZT839102"),                        # punctuation both sides
    ("two refs ZT111111 and ZT222222", "ZT111111"),       # first wins
    ("ZT8391022 is too long", None),                      # 7 digits is not a reference
    ("ZT83910 is too short", None),                       # 5 digits
    ("no reference here", None),
    ("my WIZT839102 code", None),                         # not at a word boundary
])
def test_reference(text, expected):
    assert extract_reference(text) == expected


# --- amounts ------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("I sent ₦85,000 yesterday", "85000.00"),
    ("I sent N85,000 yesterday", "85000.00"),
    ("sent NGN85000 today", "85000.00"),
    ("sent NGN 85,000.00 today", "85000.00"),
    ("sent n85000 today", "85000.00"),                    # lower case
    ("dem debit me 85k", "85000.00"),                     # Pidgin, k suffix
    ("na 85K I send", "85000.00"),                        # Pidgin, capital K
    ("sent ₦85k", "85000.00"),                            # currency + k
    ("sent ₦1.5k", "1500.00"),                            # decimal k
    ("sent N2 million", "2000000.00"),
    ("sent 2 million naira", "2000000.00"),
    ("I paid 85,000 and it failed", "85000.00"),          # comma-grouped, no currency
    ("sent N85,000 and my balance is N5,000", "85000.00"),  # first amount wins
    ("sent N500.50", "500.50"),
    ("I was in 5 queues for a long time", None),          # 'n' inside a word, bare 5
    ("the transfer has not gone", None),
    ("my number is 08031234567", None),                   # phone is not an amount
])
def test_amount(text, expected):
    assert extract_amount(text) == expected


# --- durations ----------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("waiting for 20 minutes", 20),
    ("waiting 20 mins", 20),
    ("waiting 20min", 20),
    ("waiting for 20 minuts", 20),                        # typo
    ("e don take 2 hours", 120),                          # Pidgin
    ("waiting 2hrs now", 120),
    ("it has been 1 hr", 60),
    ("waiting for 3 days", 4320),
    ("it is 1.5 hours now", 90),
    ("waiting for an hour", 60),
    ("waiting for half an hour", 30),
    ("waiting for two days", 2880),
    ("a minute ago I sent it, still nothing for 3 hours", 1),   # earliest mention wins
    ("it has not come since morning", None),
    ("I have 20 naira left", None),
])
def test_elapsed_minutes(text, expected):
    assert extract_elapsed_minutes(text) == expected


# --- institutions -------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("sent to Bank C", "BANK_C"),
    ("sent to BANK_C", "BANK_C"),
    ("sent to bank-c", "BANK_C"),
    ("sent to BankC", "BANK_C"),
    ("na Bank L I send am go", "BANK_L"),                 # Pidgin
    ("sent to bank c.", "BANK_C"),                        # lower case, end punctuation
    ("I went to the bank a few minutes ago", None),       # 'bank a' inside an English phrase
    ("sent to Bank A and Bank B", "BANK_A"),              # first wins
    ("sent to Bank Z", None),                             # not in the alias list
    ("my bank is slow", None),
])
def test_institution(text, expected):
    assert extract_institution(text, CFG) == expected


# --- masking ------------------------------------------------------------------------
def test_mask_phone_plain():
    masked, c = mask_pii("call 08031234567 now")
    assert masked == "call [PHONE] now" and c["phones"] == 1


def test_mask_phone_international_with_spaces():
    masked, c = mask_pii("call +234 803 123 4567 now")
    assert "[PHONE]" in masked and c["phones"] == 1 and "803" not in masked


def test_mask_account_number():
    masked, c = mask_pii("account 0123456789 was debited")
    assert masked == "account [ACCOUNT] was debited" and c["accounts"] == 1


def test_mask_email():
    masked, c = mask_pii("mail ada.o+x@example.com please")
    assert masked == "mail [EMAIL] please" and c["emails"] == 1


def test_mask_reference_keeps_last_three_digits():
    masked, _ = mask_pii("ref ZT839102 and zt-111222")
    assert masked == "ref ZT***102 and ZT***222"


def test_mask_leaves_amounts_alone():
    masked, c = mask_pii("sent N85,000")
    assert masked == "sent N85,000" and sum(c.values()) == 0


def test_large_naira_amount_is_not_mistaken_for_an_account():
    """NGN 1500000000 has 10 digits: it must be extracted as an amount before any masking."""
    r = extract("I sent NGN 1500000000 to Bank C", CFG)
    assert r.extracted["amount"] == "1500000000.00"


def test_no_raw_pii_survives_in_masked_message():
    msg = ("Hi I am Ada, phone 0803 123 4567, acct 0123456789, mail ada@x.com, "
           "ref ZT839102, sent N85,000 to Bank C")
    r = extract(msg, CFG)
    for leaked in ("0803", "4567", "0123456789", "ada@x.com", "839102"):
        assert leaked not in r.masked_message
    assert r.extracted["pii_masked"] == {"phones": 1, "accounts": 1, "emails": 1}


# --- language tag -------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("My transfer failed and I was debited.", "en"),
    ("My transfer no go.", "pcm"),                        # PRD Table 3 example
    ("Dem never return my money", "pcm"),                 # PRD Table 3 example
    ("Wetin happen to my money?", "pcm"),
    ("Abeg refund my money", "pcm"),
    ("E still dey pending since morning", "pcm"),
    ("Dem debit me but the person never get am", "pcm"),
    ("I don't know why it failed", "en"),                 # "don't" is not the Pidgin "don"
    ("مرحبا لم يصل المال", "other"),
])
def test_language_tag(text, expected):
    assert tag_language(text, CFG) == expected


# --- normalise ----------------------------------------------------------------------
def test_normalise_collapses_whitespace():
    assert normalise("  my   transfer\n\n failed  ") == "my transfer failed"


def test_normalise_strips_quoted_reply():
    assert normalise("My transfer failed\n> earlier text\n> more quoted") == "My transfer failed"


def test_normalise_strips_signature_and_sent_from():
    assert normalise("Please help\n-- \nAda\nSent from my iPhone") == "Please help"


def test_normalise_nfkc_converts_fullwidth_digits():
    assert normalise("ref ZT８３９１０２") == "ref ZT839102"


def test_normalise_never_returns_empty():
    assert normalise("> only a quote") == "> only a quote"


# --- full pipeline (Listing 3 shape) ------------------------------------------------
def test_listing3_shape_and_values():
    r = extract("Abeg I send N85,000 to Bank C, ref ZT839102, e don take 20 minutes", CFG)
    assert r.extracted == {
        "reference": "ZT839102", "amount": "85000.00", "elapsed_minutes": 20,
        "institution_mentioned": "BANK_C", "lang": "pcm",
        "pii_masked": {"phones": 0, "accounts": 0, "emails": 0}}


def test_everything_null_for_an_empty_enquiry():
    r = extract("How do I check my transfer limit?", CFG)
    e = r.extracted
    assert e["reference"] is None and e["amount"] is None
    assert e["elapsed_minutes"] is None and e["institution_mentioned"] is None
