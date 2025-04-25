import re

def clean_text(text):
    """
    Cleans the extracted text by removing unwanted characters, extra spaces, and irrelevant lines.
    """
    # Remove excessive whitespace and special characters
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\r\n\t]", " ", text)  # Remove line breaks, tabs
    return text

def extract_financial_data(text):
    """
    Extracts financial data such as stock prices, percentage changes, revenue, profit, valuation, and sector details.
    """
    # Regex patterns for financial data
    stock_price_pattern = r"(\b\d{1,5}\.\d{1,2}\s?(?:USD|EUR|INR|₹|\$|€)\b)"  # Fixed: No backslash before ₹
    percentage_pattern = r"([-+]?\d{1,3}\.\d{1,2}%)"
    revenue_pattern = r"(?:revenue|sales)\s(?:rose|increased|decreased|was|stood at)\s(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|crore)?)"
    profit_pattern = r"(?:profit|net income)\s(?:stood at|rose|declined|was|reported)\s(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|crore)?)"
    valuation_pattern = r"(?:valuation|market cap)\s(?:stood at|is valued at)\s(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|crore)?)"
    sector_pattern = r"(?:sector|industry):\s*([\w\s]+)"

    # Extract values
    stock_prices = re.findall(stock_price_pattern, text)
    percentages = re.findall(percentage_pattern, text)
    revenues_raw = re.findall(revenue_pattern, text)
    profits_raw = re.findall(profit_pattern, text)
    valuations_raw = re.findall(valuation_pattern, text)
    sectors = re.findall(sector_pattern, text)

    # Convert extracted numbers to a structured format
    def convert_to_float(value):
        """Converts extracted financial values to float, handling 'million', 'billion', 'crore'."""
        multiplier = 1
        if "million" in value:
            multiplier = 1e6
        elif "billion" in value:
            multiplier = 1e9
        elif "crore" in value:
            multiplier = 1e7
        return float(re.sub(r"[^\d.]", "", value)) * multiplier

    revenues = [convert_to_float(val) for val in revenues_raw]
    profits = [convert_to_float(val) for val in profits_raw]
    valuations = [convert_to_float(val) for val in valuations_raw]

    # Format extracted data
    extracted_data = {
        "Stock Prices": stock_prices,
        "Percentage Changes": percentages,
        "Revenue (in USD)": revenues,
        "Profit (in USD)": profits,
        "Valuation (in USD)": valuations,
        "Sector": sectors
    }

    return extracted_data

def process_text(text):
    """
    Cleans the text, extracts financial data, and returns structured content.
    """
    cleaned_text = clean_text(text)
    financial_data = extract_financial_data(cleaned_text)

    return {
        "cleaned_text": cleaned_text,
        "financial_data": financial_data
    }
