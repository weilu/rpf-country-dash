# RPF-Country-Dash
Dash app for displaying visualizations fro RPF project

## Development

First `git clone` this repo.

Then generate and obtain your access token from databricks following this instruction: https://docs.databricks.com/en/dev-tools/auth/pat.html
Get connection details for the SQL Warehouse compute resource which is used for providing the database connection: https://docs.databricks.com/en/integrations/compute-details.html

Export the obtained information to the following environment variables:

```bash
export ACCESS_TOKEN="[access token]"
export SERVER_HOSTNAME="[server hostname, e.g. adb-12345678.12.azuredatabricks.net"
export HTTP_PATH="[http path, e.g. /sql/1.0/warehouses/abcdxxx123]"
# For M2 Max Chip
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```


Then to setup and verify the app works locally:

```bash
pip install -r requirements.txt
python app.py
open http://127.0.0.1:8050/
```
You should see the data app.



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
