import os
import sys
import asyncio
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from vectorstore import build_faiss_index, load_faiss_index
from text_processing import process_text
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import backoff
from bs4 import BeautifulSoup
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
import re
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if not mistral_api_key:
    st.error("🚨 MISTRAL_API_KEY not found!")
    raise ValueError("MISTRAL_API_KEY missing!")

st.title("📊 RockyBot: Ultimate Equity Analysis Tool")
st.sidebar.title("📰 Equity News URLs")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "live_data" not in st.session_state:
    st.session_state.live_data = None

# URL input
urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]
urls = [url.strip() for url in urls if url.strip()]
process_url_clicked = st.sidebar.button("Analyze URLs")

# Stock ticker input
st.sidebar.subheader("📈 Live Stock Data")
ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", help="Enter a ticker symbol (e.g., 'MRF.NS' for MRF Ltd)")
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo"], index=1)
fetch_live_data = st.sidebar.button("Fetch Live Stock Data")


def get_article_content(url):
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(10000)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(5000)

            try:
                content = page.inner_text(".article-cont", timeout=30000)
            except PlaywrightTimeoutError:
                soup = BeautifulSoup(page.content(), "html.parser")
                article = soup.select_one(".article-cont") or soup
                content = article.get_text(separator="\n", strip=True)

            browser.close()
            return content.strip()
    except Exception as e:
        st.error(f"❌ Error fetching {url}: {str(e)}")
        return None


@st.cache_data(ttl=300)
@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def get_live_stock_data(ticker_input, period="5d"):
    logger.info(f"Fetching live data for {ticker_input}")
    ticker = ticker_input.upper()
    try:
        stock = yf.Ticker(ticker)
        quote = stock.info
        hist = stock.history(period=period)
        if hist.empty:
            logger.warning(f"No historical data available for {ticker}")
            return None
        previous_close = quote.get("regularMarketPreviousClose", 0)
        current_price = quote.get("regularMarketPrice", 0)
        percent_change = ((current_price - previous_close) / previous_close * 100) if previous_close else None
        currency = quote.get("currency", "USD")
        predicted_price = predict_stock_price(hist["Close"].tolist())
        return {
            "ticker": ticker,
            "current_price": current_price,
            "previous_close": previous_close,
            "volume": quote.get("regularMarketVolume", None),
            "percent_change": percent_change,
            "currency": currency,
            "predicted_price": predicted_price
        }
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
        return None


def predict_stock_price(historical_prices):
    try:
        if len(historical_prices) < 3:
            logger.warning("Not enough data for prediction")
            return None
        model = ARIMA(historical_prices, order=(1, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast[0]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None


# Analyze URLs
if process_url_clicked:
    if not urls:
        st.error("🚨 Enter at least one URL!")
    else:
        st.write("🔄 Analyzing Equity Articles...")
        articles = []
        financial_data_list = []

        for url in urls:
            raw_content = get_article_content(url)
            if raw_content:
                processed = process_text(raw_content)
                if isinstance(processed, dict) and "cleaned_text" in processed:
                    articles.append({"source": url, "content": processed["cleaned_text"]})
                    financial_data_list.append({
                        "source": url,
                        "data": processed.get("financial_data", {})
                    })

                    st.subheader(f"📊 Financial Data from {url}")
                    for key, values in processed["financial_data"].items():
                        if values:
                            if isinstance(values[0], (int, float)):
                                st.write(f"{key}: {', '.join(str(v) for v in values)}")
                            else:
                                st.write(f"{key}: {', '.join(values)}")

        # Visualization
        if financial_data_list:
            st.subheader("📈 Financial Data Visualization")
            stock_prices, revenues, profits, valuations, labels = [], [], [], [], []

            for item in financial_data_list:
                source = item["source"]
                data = item["data"]

                if data.get("Stock Prices"):
                    try:
                        value = float(re.sub(r"[^\d.]", "", data["Stock Prices"][0]))
                        stock_prices.append(value)
                        labels.append(f"Stock - {source}")
                    except:
                        continue

                if data.get("Revenue (in USD)"):
                    revenues.append(data["Revenue (in USD)"][0])
                    labels.append(f"Revenue - {source}")

                if data.get("Profit (in USD)"):
                    profits.append(data["Profit (in USD)"][0])
                    labels.append(f"Profit - {source}")

                if data.get("Valuation (in USD)"):
                    valuations.append(data["Valuation (in USD)"][0])
                    labels.append(f"Valuation - {source}")

            if labels:
                fig = go.Figure()
                if stock_prices:
                    fig.add_trace(go.Bar(x=labels[:len(stock_prices)], y=stock_prices, name="Stock Price", marker_color="blue"))
                if revenues:
                    fig.add_trace(go.Bar(x=labels[:len(revenues)], y=revenues, name="Revenue", marker_color="green"))
                if profits:
                    fig.add_trace(go.Bar(x=labels[:len(profits)], y=profits, name="Profit", marker_color="orange"))
                if valuations:
                    fig.add_trace(go.Bar(x=labels[:len(valuations)], y=valuations, name="Valuation", marker_color="purple"))

                fig.update_layout(title="📊 Financial Comparison", barmode="group", xaxis_title="Source", yaxis_title="Value")
                st.plotly_chart(fig)

        if not articles:
            st.error("🚨 No article content found!")
        else:
            st.session_state.vectorstore = build_faiss_index(articles)
            st.success("✅ Articles analyzed! You can now ask questions.")

# Live stock data
if fetch_live_data and ticker_input:
    st.session_state.live_data = get_live_stock_data(ticker_input, period)
    if st.session_state.live_data:
        currency_symbol = "₹" if st.session_state.live_data["currency"] == "INR" else "$"
        st.subheader(f"📈 Live Data for {st.session_state.live_data['ticker']}")
        st.write(f"*Current Price*: {currency_symbol}{st.session_state.live_data['current_price']:.2f}")
        st.write(f"*Previous Close*: {currency_symbol}{st.session_state.live_data['previous_close']:.2f}")
        if st.session_state.live_data['percent_change'] is not None:
            st.write(f"% Change**: {st.session_state.live_data['percent_change']:.2f}%")
        else:
            st.write("% Change**: N/A (Insufficient data)")
        st.write(f"*Volume*: {st.session_state.live_data['volume']:,}")
        if st.session_state.live_data['predicted_price']:
            st.write(f"*Predicted Next Day Price*: {currency_symbol}{st.session_state.live_data['predicted_price']:.2f}")
        else:
            st.write("⚠ Insufficient data for prediction")
    else:
        st.error(f"🚨 Failed to fetch data for {ticker_input}. Ensure the ticker is correct (e.g., 'AAPL', 'MRF.NS') and try again.")

# Fallback: load saved FAISS index
if st.session_state.get("vectorstore") is None:
    st.session_state.vectorstore = load_faiss_index()


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def extract_answer_from_docs(docs, question):
    question = question.strip() or "Summarize equity highlights from the article."
    context = "\n\n".join([f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in docs])

    if not context:
        return "No equity content found to analyze.", ""

    chat_model = ChatMistralAI(model="mistral-small", api_key=mistral_api_key, temperature=0.05)
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are a world-class equity analyst. Using the context below, answer the question precisely.
        Focus on stock prices, financial metrics, and investor insights.

        Question: {question}
        Context: {context}

        Answer:
        """
    )

    prompt = prompt_template.format(question=question, context=context)
    response = chat_model.invoke(prompt)
    return response.content.strip(), context


def extract_numbers_for_chart(context):
    lines = context.splitlines()
    labels, values = [], []

    for line in lines:
        match_nifty = re.search(r"(Nifty(?: [\w]+)?) (?:at|to|below|above|around)?\s([\d,]+)", line, re.IGNORECASE)
        if match_nifty:
            label = match_nifty.group(1).strip()
            value = float(match_nifty.group(2).replace(",", ""))
            labels.append(label)
            values.append(value)

        match_basic = re.search(r"([A-Za-z\s]+):\s*₹?([\d,.]+)", line)
        if match_basic:
            labels.append(match_basic.group(1).strip())
            values.append(float(match_basic.group(2).replace(",", "")))

    return labels, values


query = st.text_input("🔍 Ask a Question About the Articles:")
ask_btn = st.button("Get Answer")

if ask_btn and query:
    if st.session_state.get("vectorstore") is None:
        st.error("🚨 Analyze at least one article first.")
    else:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(query)

        try:
            answer, context = extract_answer_from_docs(docs, query)
            st.header("📢 RockyBot's Answer")
            st.write(answer)

            if any(word in query.lower() for word in ["chart", "plot", "graph"]) and context:
                labels, values = extract_numbers_for_chart(context)
                if labels and values:
                    chart = go.Figure([go.Bar(x=labels, y=values, marker_color='darkcyan')])
                    chart.update_layout(title="📊 Chart From Your Question", xaxis_title="Metric", yaxis_title="Value")
                    st.plotly_chart(chart)
                else:
                    st.info("🤖 No chartable financial data found in the answer.")

            st.subheader("🔗 Sources Used:")
            for src in set(doc.metadata["source"] for doc in docs):
                st.write(src)

        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")
