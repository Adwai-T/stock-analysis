from kiteconnect import KiteConnect
import json

CREDS = "./credentials.json"

def main():
    with open(CREDS, "r") as f:
        c = json.load(f)

    api_key = c["API_KEY"]
    api_secret = c["API_SECRET"]

    kite = KiteConnect(api_key=api_key)

    print("Paste your request_token from login URL:")
    request_token = input("> ").strip()

    data = kite.generate_session(request_token, api_secret)
    access_token = data["access_token"]

    print("\nYour ACCESS_TOKEN is:\n")
    print(access_token)

    c["ACCESS_TOKEN"] = access_token
    with open(CREDS, "w") as f:
        json.dump(c, f, indent=2)

    print("\nSaved to credentials.json")

if __name__ == "__main__":
    main()
