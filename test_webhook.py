"""Simple webhook test client."""

import requests


WEBHOOK_URL = "http://localhost:5000/webhook"
TEST_PAYLOAD = {
    "pdf_path": "sample.pdf",
    "recipient_email": "test@example.com",
}


def main():
    """Send a test request to the local webhook endpoint."""
    try:
        response = requests.post(WEBHOOK_URL, json=TEST_PAYLOAD, timeout=30)
        print(response.json())
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the webhook server. Make sure Flask is running on port 5000.")
    except requests.exceptions.Timeout:
        print("Error: The webhook request timed out.")
    except requests.exceptions.JSONDecodeError:
        print("Error: The webhook response was not valid JSON.")
        print(response.text)
    except requests.exceptions.RequestException as exc:
        print(f"Error: Webhook request failed. {exc}")


if __name__ == "__main__":
    main()
