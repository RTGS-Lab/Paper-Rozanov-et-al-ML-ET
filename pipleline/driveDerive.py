import os
import io
import shutil
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import schedule
import time
from datetime import datetime, timedelta

SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_google_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    service = build('drive', 'v3', credentials=creds)
    return service

def download_file(service, file_id, file_name, year):
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(os.getcwd(), file_name)
    
    with io.FileIO(f'./data/MODIS/{year}/{file_name}', 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
    
    print(f"File '{file_name}' downloaded successfully.")
    return file_path


def delete_file(service, file_id):
    service.files().delete(fileId=file_id).execute()
    print("File deleted from Google Drive.")


def list_and_process_files(service):
    query = f"'0AJq-LJPFFdtXUk9PVA' in parents"
    results = service.files().list(q=query, pageSize=366, fields="files(id, name)").execute()
    files = results.get('files', [])
    
    if not files:
        print("No files found.")
    else:
        print("Files found:")
        for file in files:
            file_id = file['id']
            file_name = file['name']
            if file_name.split('.')[-1]=='tif':
                year = file_name.split('_')[1]
                print(f"Downloading: {file_name} ({file_id})")
                file_path = download_file(service, file_id, file_name, year)
                
                delete_file(service, file_id)
def job():
    print(f"Task running at {datetime.now()}")
    service = authenticate_google_drive()
    list_and_process_files(service)

if __name__ == '__main__':
    STOP_TIME = datetime(2025, 6, 1, 14, 0)

    while True:
        now = datetime.now()
        if now >= STOP_TIME:
            print(f"Stopping script at {now}")
            break
    
        start_time = datetime.now()
        job()
        
        while datetime.now() < start_time + timedelta(hours=1):
            time.sleep(10)  # Sleep in short intervals