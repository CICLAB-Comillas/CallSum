from google.cloud import secretmanager

# ID del proyecto de GCP y del secreto
SECRET_ID = "712334688886"
SECRET_NAME = "phone_calls_dataset"

def get_API_KEY() -> str:
    # Crea el cliente de Secret Manager
    client = secretmanager.SecretManagerServiceClient()

    # Construye el nombre del secreto
    secret_name = f"projects/{SECRET_ID}/secrets/{SECRET_NAME}/versions/latest"

    # Accede al secreto
    response = client.access_secret_version(request={"name": secret_name})

    # Obtiene el valor del secreto
    secret_value = response.payload.data.decode("UTF-8")

    # Utiliza el valor del secreto como necesites
    return secret_value