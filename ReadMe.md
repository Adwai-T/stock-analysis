# Stock Analysis

## Run App

`python server/app.py`

## Setup

### Downloads

1. For Scripts - `pip install pandas numpy requests tensorflow`
2. 

### File Structure

Create a file `credentials.json` in the root of the project.

```json
{
  "API_KEY": "",
  "API_SECRET": "",
  "ACCESS_TOKEN": "",
  "REQUEST_TOKEN": ""
}
```

`python scripts/generate_access_token.py` use to generate Access Token that needs to be refereshed everyday.

`REQUEST_TOKEN` is given when loggin in to kite connect manually.

Use URL initially - `https://kite.trade/connect/login?api_key=YOUR_API_KEY`.
This will give the request token.

`ACCESS_TOKEN` is recieved in the Url and needs to be update when login in is completed.