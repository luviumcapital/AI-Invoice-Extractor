import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import parse_invoice_fields

def test_parse_invoice_fields():
    """Test the parse_invoice_fields function."""
    sample = "Invoice #12345\nDate: 01/01/2024\nTotal: $123.45\nFrom: ACME Corp"
    fields = parse_invoice_fields(sample)
    assert fields["Invoice Number"] == "Invoice #12345"
    assert fields["Date"] == "Date: 01/01/2024"
    assert fields["Total Amount"] == "Total: $123.45"
    assert fields["Vendor"] == "From: ACME Corp"