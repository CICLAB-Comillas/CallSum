import argparse
import json
import time
import os
from random import choice, randint
from typing import Dict, List

import datetime
import openai
import pandas as pd
from pandas import DataFrame
from rich.console import Group
from rich.live import Live
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TextColumn, TimeElapsedColumn, TimeRemainingColumn)
from rich.rule import Rule

from secret_key import get_API_KEY

### ----------------------------------ARGUMENTOS---------------------------------------------

# Ruta de este archivo
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

### Argumentos
parser = argparse.ArgumentParser()

# Api Key
parser.add_argument('-k','--api_key', help='OpenAI Api key', default=get_API_KEY())

# Número de llamadas a generar
parser.add_argument('-n', '--num_iter',help='Nº llamadas a generar', required=True)

# Path del CSV
parser.add_argument('-p','--path', help='Ruta para guardado del archivo CSV generado', default=os.path.join(FILE_PATH,"dataset.csv"))

# Path del JSON con los parámetros de la llamada
parser.add_argument('-j','--json', help='Ruta con el archivo JSON con los parámetros para las llamadas', default=os.path.join(FILE_PATH,"params.json"))

# Número de batchs
parser.add_argument('-b','--batches', help='Número de batches', default=1)

# args = parser.parse_args()

# openai.api_key = args.api_key
# N_LLAMADAS = int(args.num_iter)
# CSV_PATH = args.path
# JSON_PATH = args.json
# N_BATCHES = int(args.batches)
# N_BATCHES = N_BATCHES if N_LLAMADAS>=N_BATCHES else 1

openai.api_key = get_API_KEY()
N_LLAMADAS = 100
CSV_PATH = os.path.join(FILE_PATH,"dataset.csv")
JSON_PATH = os.path.join(FILE_PATH,"params.json")
N_BATCHES = 10
N_BATCHES = N_BATCHES if N_LLAMADAS>=N_BATCHES else 1

CSV_PATH_METRICAS = CSV_PATH.replace(".csv","_metricas.csv")

TIEMPO_ESPERA_TRAS_ERROR_LLAMADA_SEC = 5 #segundos

### ----------------------------------EXCEPCIONES---------------------------------------------
class FormatoRespuestaExcepcion(Exception):
    """Error en el formato de la respuesta que devuelve chatGPT"""
    pass

### ----------------------------------FUNCIONES---------------------------------------------

# Reading params from JSON file
with open(JSON_PATH, 'r') as f:
    PARAMS = json.load(f)

def dividir_en_batches(n_llamadas: int, n_batches: int) -> List[int]:
    llamadas_x_batch = n_llamadas // n_batches  # Calcula la parte entera de la división
    residuo = n_llamadas % n_batches  # Calcula el residuo de la división

    lotes = [llamadas_x_batch] * n_batches  # Crea una lista con el cociente repetido n_batches veces

    # Distribuye el residuo en los lotes
    for i in range(residuo):
        lotes[i] += 1

    return lotes

def generar_telf_aleatorio() -> str:
    return f"+34 {randint(6e8,75e7-1)}"

def generar_parametros_aleatorios_llamada() -> str:
    """ Devuelve un diccionario con un valor aleatorio escogido para cada parámetro """
    call_params = {}
    # Parámetros aleatorios
    for param in PARAMS["Llamada"]:
        call_params[param] = choice(PARAMS["Llamada"][param])
        
    # Parámetros fijos
    call_params['Teléfono'] = generar_telf_aleatorio()
    call_params['Dirección'] = "Calle de la cuesta 15"
    call_params['Empresa'] = "Empresa"
    return call_params

def generar_parametros_aleatorios_resumen() -> Dict[str, str]:
    """ Devuelve un diccionario con un valor aleatorio escogido para cada parámetro """
    resumen_params = {}

    # Parámetros aleatorios
    for param in PARAMS["Resumen"]:
        resumen_params[param] = choice(PARAMS["Resumen"][param])
    
    return resumen_params

def generar_prompt_llamada(call_params: Dict[str,List[str]] = None) -> str:
    """ Devuelve un prompt para generar una llamada en base a unos parámetros"""
    call_params = generar_parametros_aleatorios_llamada() if call_params is None else call_params
    return f"Genera una conversación telefónica {call_params['Conversación']} donde un cliente con teléfono {call_params['Teléfono']} y dirección {call_params['Dirección']}, con una actitud {call_params['Actitud']}, se pone en contacto con la empresa suministradora {call_params['Empresa']} para reportar un problema de {call_params['Problema']}. Al final, la llamada {call_params['Resuelta']} queda resuelta de forma {call_params['Forma']}."

def generar_prompt_resumen(resumen_params: Dict[str, str] = None) -> str:
    """ Devuelve un prompt para generar un resumen en base a unos parámetros"""
    resumen_params = generar_parametros_aleatorios_resumen() if resumen_params is None else resumen_params
    return f"Además, genera un resumen {resumen_params['Extension']} de esta conversación."

def combinar_prompts(prompt_llamada, prompt_resumen) -> str:
    """ Combinar dos prompts (conversación y resumen) en uno único para ahorrar tokens al lanzar la petición a la API de ChatGPT
    """
    return f"{prompt_llamada} {prompt_resumen}. Antes de la conversación incluye [Conversación] y antes del resumen [Resumen]."

def ask_ChatGPT(prompt:str):
    """ Sends a request message as a `prompt` to OpenAI API. Params
        * `prompt`: text input to API
        Returns the API completion (if received).
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    if 'choices' in completion:
        return completion
    else:
        return None
    
def procesar_completion(completion: str) -> Dict[str, str]:
    """ Convierte la respuesta de la API ChatGPT a un objeto `Dict`, añadiendo el prompt proporcionado"""

    dict_llamadas = {}

    try:
        # Obtener el contenido del mensaje (conversación y resumen)
        contenido = completion.choices[0].message['content']

        # Obtener el campo conversación de la respuesta
        conversacion = contenido.split('[Conversación]')[1].split('[Resumen]')[0].strip()
        conversacion = conversacion[1:] if conversacion[0] == ':' else conversacion
        conversacion = conversacion[1:] if conversacion[0] == ' ' else conversacion
        dict_llamadas["Transcripcion"] = conversacion

        # Obtener el campo resumen de la respuesta
        resumen = contenido.split("[Resumen]")[1].split("}")[0].strip()
        resumen = resumen[1:] if resumen[0] == ':' else resumen
        resumen = resumen[1:] if resumen[0] == ' ' else resumen
        dict_llamadas["Resumen"] = resumen
    except IndexError:
        raise FormatoRespuestaExcepcion()

    # Obtener el uso
    dict_metricas = dict(completion.usage)

    return (dict_llamadas, dict_metricas)

def generar_llamada(df_llamadas: DataFrame, df_metricas: DataFrame, llamada: int, batch: int) -> DataFrame:
    """ Genera una nueva fila del dataset con la conversación, resumen y el tipo de resumen
    """

    # ------------ 1 - Generando parámetros aleatorios ------------
    # Crear tarea del task
    progreso_paso_batch_id_0 = progreso_paso_batch.add_task('',action='Generando parámetros aleatorios...')

    # Parametros aleatorios llamada
    call_params = generar_parametros_aleatorios_llamada()

    # Parametros aleatorios Resumen
    resumen_params = generar_parametros_aleatorios_resumen()

    # Actualizar el progreso del batch
    progreso_paso_batch.update(progreso_paso_batch_id_0, action='[bold green]Parámetros aleatorios [✔]')
    progreso_paso_batch.stop_task(progreso_paso_batch_id_0)

    # ------------ 2 - Construyendo prompt ------------
    # Crear tarea del task
    progreso_paso_batch_id_1 = progreso_paso_batch.add_task('',action='Construyendo prompt...')

    # Generar Prompt llamada
    prompt_llamada = generar_prompt_llamada(call_params)

    # Generar Prompt resumen
    prompt_resumen = generar_prompt_resumen(resumen_params)

    # Combinar llamada y resumen en un Prompt
    prompt = combinar_prompts(prompt_llamada, prompt_resumen)

    # Actualizar el progreso del batch
    progreso_paso_batch.update(progreso_paso_batch_id_1, action='[bold green]Prompt [✔]')
    progreso_paso_batch.stop_task(progreso_paso_batch_id_1)

    # ------------ 3 - Realizando llamada API ------------
    # Crear tarea del task
    progreso_paso_batch_id_2 = progreso_paso_batch.add_task('',action='Realizando llamada API OPENAI...')

    # Inicio temporizador llamada API Openai
    inicio = time.time()

    # Tiempo inicial de espera cuando se produce un error de conexión
    tiempo_espera = TIEMPO_ESPERA_TRAS_ERROR_LLAMADA_SEC

    while True:
        try:
            # Realizar llamada a la API de ChatGPT
            completion = ask_ChatGPT(prompt) # Respuesta
            break
        except openai.error.APIError: 
            # Si hay un error espera un tiempo determinado e intenta de nuevo la llamada
            print(f"Conection error. Waiting for server...")
            time.sleep(tiempo_espera) # Espera

            # Incrementa el siguiente tiempo de espera
            tiempo_espera += TIEMPO_ESPERA_TRAS_ERROR_LLAMADA_SEC

    # Fin temporizador
    tiempo = round(time.time()-inicio, 2)

    # Actualizar el progreso del batch
    progreso_paso_batch.update(progreso_paso_batch_id_2, action='[bold green]Llamada API OPENAI [✔]')
    progreso_paso_batch.stop_task(progreso_paso_batch_id_2)

    # ------------ 4 - Procesando respuesta ------------
    # Crear tarea del task
    progreso_paso_batch_id_3 = progreso_paso_batch.add_task('',action='Procesando respuesta...')

    try:
        # Obtener la conversación y el resumen en un diccionario
        dict_llamada, dict_metricas = procesar_completion(completion)
    except FormatoRespuestaExcepcion:
        # Eliminar las respuestas
        progreso_paso_batch.remove_task(progreso_paso_batch_id_0)
        progreso_paso_batch.remove_task(progreso_paso_batch_id_1)
        progreso_paso_batch.remove_task(progreso_paso_batch_id_2)
        progreso_paso_batch.remove_task(progreso_paso_batch_id_3)
        raise

    # Añadir el resto de campos
    for param in list(PARAMS['Llamada'].keys()):
        # Añadir el parametro de la llamada
        dict_llamada[param] = call_params[param]
    
    for param in list(PARAMS['Resumen'].keys()):
        # Añadir el parametro del resumen
        dict_llamada[param] = resumen_params[param]

    dict_metricas["time"] = tiempo # Añadir el tiempo que ha tardado la petición
    dict_metricas["llamada"] = llamada
    dict_metricas["batch"] = batch

    # Crear la fila de llamada
    fila_llamada = pd.Series(dict_llamada).to_frame().T

    # Concatenar la nueva fila al final del df
    df_llamadas = pd.concat([df_llamadas, fila_llamada], ignore_index=True)

    # Crear la fila de llamada
    fila_metricas = pd.Series(dict_metricas).to_frame().T
    fila_metricas = fila_metricas.astype(dtype={'prompt_tokens': "int64",'completion_tokens': "int64",'total_tokens': "int64",'time': "float64",'llamada': "int64",'batch': "int64"})

    # Concatenar la nueva fila al final del df
    df_metricas = pd.concat([df_metricas, fila_metricas], ignore_index=True)

    # Actualizar el progreso del batch
    progreso_paso_batch.update(progreso_paso_batch_id_3, action='[bold green]Llamada API OPENAI [✔]')
    progreso_paso_batch.stop_task(progreso_paso_batch_id_3)

    # Eliminar las respuestas
    progreso_paso_batch.remove_task(progreso_paso_batch_id_0)
    progreso_paso_batch.remove_task(progreso_paso_batch_id_1)
    progreso_paso_batch.remove_task(progreso_paso_batch_id_2)
    progreso_paso_batch.remove_task(progreso_paso_batch_id_3)
    
    return df_llamadas, df_metricas

def guardar_CSV(df: DataFrame, path: str = CSV_PATH) -> None:
    """ Guarda el dataframe indicado como un archivo CSV. Si el archivo CSV ya existe, entonces"""
    if os.path.exists(path):
        # Leer el CSV y obtener el último índice
        df_csv = pd.read_csv(path,sep=";") 
        try:
            ultimo_indice = df_csv.iloc[-1]["index"]
        except IndexError:
            ultimo_indice = -1

        # Añadir una columna de indices´
        df.insert(loc=0, column='index', value=pd.RangeIndex(start=ultimo_indice+1, stop=ultimo_indice+len(df)+1, step=1))

        # Añadir el DF al CSV
        df.to_csv(path, header=False, index=False, mode='a', sep=';', encoding='utf-8')
    else:
        # Añadir una columna de indices
        df.insert(loc=0, column='index', value=pd.RangeIndex(start=0, stop=len(df), step=1))

        # Crear el CSV
        df.to_csv(path, header=True, index=False, index_label="index", sep=';', encoding='utf-8')

def actualizar_tiempo_restante(tiempo_acumulado: int, llamadas_completadas: int, llamadas_restantes: int) -> str:
    """ Calcula el tiempo restante en el formato h:mm:ss"""
    tiempo_restante_sec = llamadas_restantes*int(tiempo_acumulado/llamadas_completadas) if llamadas_completadas else '-:--:--'
    tiempo_restante_sec = str(datetime.timedelta(seconds = tiempo_restante_sec))
    return tiempo_restante_sec

### ----------------------------------PROGRESS BARS---------------------------------------------

# Barra de progreso total
progreso_total = Progress(
    TextColumn('[bold blue]Progreso dataset: [bold purple]{task.percentage:.0f}% ({task.completed}/{task.total}) '),
    BarColumn(),
    TimeElapsedColumn(),
    TextColumn('tiempo restante: [bold cyan]{task.fields[name]}'),
    TextColumn('{task.description}'),
)

# Barra de progreso total del batch
progreso_total_batch = Progress(
    TimeElapsedColumn(),
    TextColumn('[bold blue] Batch {task.fields[name]}'),
    BarColumn(),
    TextColumn('({task.completed}/{task.total}) llamadas generadas')
)

# Barra de progreso paso del batch
progreso_paso_batch = Progress(
    TextColumn('  '),
    TimeElapsedColumn(),
    TextColumn('{task.fields[action]}'),
    SpinnerColumn('dots'),
)

# Grupo de todas las barras de progreso a renderizar
group = Group(
    progreso_total,
    Rule(style='#AAAAAA'),
    progreso_total_batch,
    progreso_paso_batch
)

# Autorender para el grupo de barras
live = Live(group)

if __name__ == "__main__":

    # Reparto equilibrado de las llamada en Batches
    batches = dividir_en_batches(N_LLAMADAS,N_BATCHES)

    # Crear Dataframe
    df_llamadas = DataFrame(columns=['Transcripcion','Resumen']+list(PARAMS['Llamada'].keys())+list(PARAMS['Resumen'].keys()))

    df_metricas = DataFrame(columns=['prompt_tokens','completion_tokens','total_tokens','time','llamada','batch'])

    # Tarea del progreso total
    progreso_total_id = progreso_total.add_task(description=f'[bold #AAAAA](batch {0} de {N_BATCHES})', total=N_LLAMADAS,name = '-:--:--')

    with live:
        try:
            # Se empieza a contar el tiempo del proceso 
            tiempo_inicial_sec = time.time()

            for batch, llamadas_batch in enumerate(batches):
                # Actualizar el número de batch
                progreso_total.update(progreso_total_id, description=f'[bold #AAAAA](batch {batch+1} de {N_BATCHES})')
                
                # Crear tarea del task de batch completo
                progreso_total_batch_id = progreso_total_batch.add_task('',total=llamadas_batch, name=batch)

                for llamada in range(llamadas_batch):
                    try:
                        # Generar una llamada
                        df_llamadas, df_metricas = generar_llamada(df_llamadas, df_metricas, llamada, batch)
                    except FormatoRespuestaExcepcion:
                        llamada -= -1

                    # Actualizar el número de llamadas
                    progreso_total_batch.update(progreso_total_batch_id, advance=1)

                    # Actualizar el progreso total
                    progreso_total.update(progreso_total_id, advance=1, refresh=True)

                    # Actualizar tiempo restante
                    tiempo_acumulado_sec = int(time.time()-tiempo_inicial_sec)
                    llamadas_restantes = N_LLAMADAS - progreso_total._tasks[progreso_total_id].completed
                    progreso_total.update(progreso_total_id, name = actualizar_tiempo_restante(tiempo_acumulado_sec, llamada+1, llamadas_restantes))
                # Guarda el df de llamadas en un csv
                guardar_CSV(df_llamadas.iloc[-llamadas_batch:])
                guardar_CSV(df_metricas.iloc[-llamadas_batch:], CSV_PATH_METRICAS) # Guardar métricas

        except Exception as e: # work on python 3.x
            print(f"Se ha producido un error: \n" + str(e))
            # Guarda el df de llamadas en un csv
            guardar_CSV(df_llamadas.iloc[-llamadas_batch:])
            guardar_CSV(df_metricas.iloc[-llamadas_batch:], CSV_PATH_METRICAS) # Guardar métricas