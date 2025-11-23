#!/usr/bin/env python3
"""
Download files from Google Drive folder into a local directory.

Simple script using Google Drive API v3.
Supports either user OAuth (client_secret.json) or service account key.

Example:
python tools/download_from_drive.py --drive-folder EO_Exports --out-dir data/raw --pattern "india_*_before_*.tif"
"""
import argparse, os, fnmatch, io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from tqdm import tqdm

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = ".drive_token.json"

def get_service_user(client_secret):
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secret or not os.path.exists(client_secret):
                raise FileNotFoundError("client_secret.json not found; provide --client-secret or use --service-account-key")
            flow = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def get_service_sa(sa_key):
    creds = service_account.Credentials.from_service_account_file(sa_key, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def find_folder(drive, folder_name):
    q = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    res = drive.files().list(q=q, spaces="drive", fields="files(id,name)", pageSize=10).execute()
    files = res.get("files", [])
    if not files:
        return None
    return files[0]["id"]

def list_files(drive, folder_id):
    files=[]
    page_token=None
    q = f"'{folder_id}' in parents and trashed = false"
    while True:
        resp = drive.files().list(q=q, spaces="drive", fields="nextPageToken, files(id,name,size)", pageSize=1000, pageToken=page_token).execute()
        files.extend(resp.get("files",[]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def download(drive, file_id, name, out_dir):
    request = drive.files().get_media(fileId=file_id)
    out_path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    fh = io.FileIO(out_path, "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024*8)
    done=False
    with tqdm(unit="chunk", desc=name) as pbar:
        while not done:
            status, done = downloader.next_chunk()
            if status:
                pbar.update(1)
    fh.close()
    return out_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--service-account-key", default=None)
    p.add_argument("--client-secret", default=None)
    p.add_argument("--folder-id", default=None)
    p.add_argument("--drive-folder", default="EO_Exports")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pattern", default="*.tif")
    args = p.parse_args()

    if args.service_account_key:
        drive = get_service_sa(args.service_account_key)
    else:
        drive = get_service_user(args.client_secret)
    folder_id = args.folder_id or find_folder(drive, args.drive_folder)
    if not folder_id:
        raise SystemExit("Drive folder not found")
    files = list_files(drive, folder_id)
    matches = [f for f in files if fnmatch.fnmatch(f["name"], args.pattern)]
    if not matches:
        print("No matching files.")
        return
    for f in matches:
        print("Downloading", f["name"])
        download(drive, f["id"], f["name"], args.out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
