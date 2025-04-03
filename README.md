# RPF-Country-Dash
Dash app for displaying visualizations fro RPF project

## Development

First `git clone` this repo.

Then prepare .env: `cp .env.example .env`

Then generate and obtain your access token from databricks following this instruction: https://docs.databricks.com/en/dev-tools/auth/pat.html
Get connection details for the SQL Warehouse compute resource which is used for providing the database connection: https://docs.databricks.com/en/integrations/compute-details.html

Export the obtained information to the following environment variables, update them in .env file:

```bash
ACCESS_TOKEN="[access token]"
SERVER_HOSTNAME="[server hostname, e.g. adb-12345678.12.azuredatabricks.net"
HTTP_PATH="[http path, e.g. /sql/1.0/warehouses/abcdxxx123]"
```

Then to setup and verify the app works locally:

```bash
pip install -r requirements.txt
dotenv run -- python app.py
open http://127.0.0.1:8050/
```
You should see the data app.

### Tests

To run the python unit tests:

```
python -m unittest discover tests/
```

Make sure all tests pass locally before sending a PR.

## Development within docker container
1. Edit .env to update your environment variables after copying the sample env file. (Do not use quotations around the values)

```bash
cp .env.example .env
```

2. Run the following commands

```bash
docker build . --tag dash
docker run -p 8080:8080 -v ./:/dash-app  --env-file .env dash
```

## Deployment

To deploy a version of the app with authentication enabled, post deployment set the following env vars:
```
AUTH_ENABLED=1
SECRET_KEY=yoursecretkey
```

The app will read usernames and salted passwords from the database, so be sure to configure them there: see `QueryService.get_user_credentials`. You may use [scripts/hash_password.py](scripts/hash_password.py) to hash passwords.
