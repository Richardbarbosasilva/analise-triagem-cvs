#!/usr/bin/env python3
# mvp_triagem_cv_debug.py
# Versão com logs e "prints" detalhados para troubleshooting em todos os passos

import os
import sys
import re
import io
import json
import time
import shutil
import logging
import hashlib
import requests
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
import unicodedata
import pdfplumber
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
from email.message import EmailMessage
from email.utils import make_msgid
import smtplib
import mariadb
from logging.handlers import RotatingFileHandler

# -----------------------
# Config inicial / DEBUG
# -----------------------

DEBUG = os.environ.get('DEBUG', 'true').lower() in ('1','true','yes')
LOG_DIR = os.environ.get('LOG_DIR', '/var/www/backend_pipeline_analise_cv_sistema_recrutamento_imais/vagas_description/attachments/triagem_cv_logs')
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Logging: console + arquivo rotativo
logger = logging.getLogger('triagem_cv')
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)
fh = RotatingFileHandler(os.path.join(LOG_DIR, 'triagem_cv.log'), maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
fh.setFormatter(fmt)
logger.addHandler(fh)

logger.info('Iniciando script de triagem automática de currículos (modo DEBUG=%s)...', DEBUG)

date= datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print (f"Script iniciado em {date}")

# -----------------------
# Config (use ENV vars or edite defaults)
# -----------------------
print ("carregando configurações e variáveis de ambiente...")

DB_HOST = os.environ.get("DB_HOST", "<ip>")
DB_PORT = int(os.environ.get("DB_PORT", "<port>"))
DB_NAME = os.environ.get("DB_NAME", "painel_wp")
DB_USER = os.environ.get("DB_USER", "<user>")
DB_PASS = os.environ.get("DB_PASS", "<password>")  # em produção use secrets

TABLE_NAME = os.environ.get("FORMINATOR_FORMS", "wp_frmt_form_entry_meta")

# crie um arquivo chamado parametros_vagas.json

JOB_REQS_JSON = os.environ.get("JOB_REQS_JSON", "parametros_vagas.json")
ATTACHMENTS_DIR = os.environ.get("ATTACHMENTS_DIR", "/tmp/triagem_cv_attachments")
LOGO_PATH = os.environ.get("LOGO_PATH", "")
TOP_K = int(os.environ.get("TOP_K", "10"))

# SMTP google

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER", "<email_send>")
SMTP_PASS = os.environ.get("SMTP_PASS", "<app google account password>")

REPORT_TO = os.environ.get("REPORT_TO", "<email_receive>")
REPORT_CC = os.environ.get("REPORT_CC", "<email_send_cc>")

# Timeout para downloads

DOWNLOAD_TIMEOUT = int(os.environ.get('DOWNLOAD_TIMEOUT', '15'))

ensure_dirs = [ATTACHMENTS_DIR, LOG_DIR]
for d in ensure_dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

# -----------------------
# Utilitários de debug (salvam artefatos)
# -----------------------

print ("carregando utilitários de debug...")

def debug_save_text_artifact(basename, content):
    try:
        p = Path(ATTACHMENTS_DIR) / f"debug_{basename}.txt"
        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.debug('Salvo artefato texto: %s', p)
        return str(p)
    except Exception as e:
        logger.exception('Falha ao salvar artefato texto: %s', e)
        return None


def debug_save_binary_artifact(basename, data_bytes):
    try:
        p = Path(ATTACHMENTS_DIR) / f"debug_{basename}"
        with open(p, 'wb') as f:
            f.write(data_bytes)
        logger.debug('Salvo artefato binário: %s', p)
        return str(p)
    except Exception as e:
        logger.exception('Falha ao salvar artefato binário: %s', e)
        return None

# -----------------------
# Funções utilitárias
# -----------------------

print ("carregando funções utilitárias...")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def file_hash(path):
    h = hashlib.sha1()
    try:
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        logger.exception('Erro ao calcular hash de %s', path)
        return None


def normalize_text(s):
    s = (s or '')
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# -----------------------
# Conexão MariaDB
# -----------------------

print ("Inicializando conexão com o banco de dados MariaDB...")

def connect_db():
    try:
        conn = mariadb.connect(
            host=DB_HOST, port=DB_PORT,
            user=DB_USER, password=DB_PASS,
            database=DB_NAME,
            autocommit=True
        )
        cur = conn.cursor()
        logger.info("Conectado ao MariaDB: %s@%s:%s/%s", DB_USER, DB_HOST, DB_PORT, DB_NAME)

        # testar leitura
        try:
            cur.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 1;")
            record = cur.fetchone()
            logger.debug('Conexão bem sucedida com a tabela %s, extraindo primeiras informações da base de dados: %s', TABLE_NAME, record)
        except Exception as e:
            logger.warning('Não foi possível consultar a tabela %s (detalhe: %s)', TABLE_NAME, e)

        return conn, cur
    except mariadb.Error as e:
        logger.exception("Erro ao conectar ao MariaDB: %s", e)
        raise

logger.debug('Função connect_db definida')

# -----------------------
# Parse upload meta (serializado PHP ou string contendo URL)
# -----------------------

print ("Carregando função de parsing de metadados de upload...")

def parse_upload_meta(meta_value):
    logger.debug('parse_upload_meta: recebendo meta com tamanho %s', len(str(meta_value)) if meta_value else 0)
    if not meta_value:
        return None
    s = str(meta_value)
    m = re.search(r'file_url\"\;s:\d+:\"([^\"]+)\"', s)
    if m:
        url = m.group(1)
        logger.debug('parse_upload_meta: encontrou URL via serialized php: %s', url)
        return url
    m2 = re.search(r'(https?://[^\s\"\']+)', s)
    if m2:
        url = m2.group(1)
        logger.debug('parse_upload_meta: encontrou URL via regex geral: %s', url)
        return url
    logger.debug('parse_upload_meta: nenhuma URL encontrada')
    return None

# -----------------------
# Download de arquivos via HTTP
# -----------------------

print ("Carregando função para fazer download de arquivos pdf via conexão http...")

def download_file(url, dest_dir, prefix=None):
    ensure_dir(dest_dir)
    local_name = url.split('/')[-1] or f'download_{int(time.time())}'
    if prefix:
        local_name = f"{prefix}_{local_name}"
    local_path = os.path.join(dest_dir, local_name)
    logger.info('download_file: iniciando download %s -> %s', url, local_path)
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info('Baixado: %s -> %s', url, local_path)
        debug_save_binary_artifact(f"last_download_{local_name}", open(local_path,'rb').read())
        return local_path
    except Exception as e:
        logger.exception('Falha ao baixar %s: %s', url, e)
        return None

# -----------------------
# Extração de texto (PDF + imagem OCR fallback)
# -----------------------

print ("Carregando funções para extração de texto de PDFs")

def extract_text_from_pdf(path):
    text = ''
    try:
        with pdfplumber.open(path) as pdf:
            logger.debug('extract_text_from_pdf: paginas detectadas=%s para %s', len(pdf.pages), path)
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    ptext = page.extract_text() or ''
                    text += ptext + '\n'
                    # salvar primeira página como imagem para debug
                    if i == 1:
                        try:
                            im = page.to_image(resolution=150)
                            img_bytes = im.original.bytes
                            debug_save_binary_artifact(f'{Path(path).stem}_page1.png', img_bytes)
                        except Exception:
                            logger.debug('Não foi possível salvar primeira página como imagem para %s', path)
                except Exception as e:
                    logger.exception('Erro extraindo texto da página %s de %s: %s', i, path, e)
    except Exception as e:
        logger.exception('pdfplumber erro para %s: %s', path, e)
    return text

print ("Carregando função principal de extração de texto (PDF + OCR)...")

def extract_text(path):
    ext = Path(path).suffix.lower()
    text = ''
    logger.debug('extract_text: tentando extrair texto de %s (ext=%s)', path, ext)
    if ext == '.pdf':
        text = extract_text_from_pdf(path)
        if text.strip():
            debug_save_text_artifact(f'{Path(path).stem}_extracted', text[:5000])
            return text
    # fallback: tentar abrir como imagem e OCR
    try:
        im = Image.open(path)
        logger.debug('extract_text: arquivo aberto como imagem, executando OCR %s', path)
        ocr_text = pytesseract.image_to_string(im)
        debug_save_text_artifact(f'{Path(path).stem}_ocr', ocr_text[:5000])
        return ocr_text
    except Exception as e:
        logger.debug('extract_text: não é imagem ou OCR falhou (%s): %s', path, e)
    return text or ''

# -----------------------
# Carregar requisitos de vaga (JSON)
# -----------------------

print ("Carregando função de requisitos de vaga a partir dos parâmetros JSON...")

json_path_default = os.path.join('/var/www/backend_pipeline_analise_cv_sistema_recrutamento_imais/vagas_description/parametros_cargos', JOB_REQS_JSON)

def load_job_requirements(json_path=None):
    path = json_path or json_path_default
    logger.info('load_job_requirements: carregando %s', path)
    if not path or not os.path.exists(path):
        logger.warning('Job requirements JSON não encontrado em %s', path)
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.debug('Job requirements carregado: %s chaves', len(data))
            debug_save_text_artifact('job_requirements_preview', json.dumps(data, indent=2)[:5000])
            return data
    except Exception as e:
        logger.exception('Falha ao carregar JSON de requisitos: %s', e)
        return {}

# -----------------------
# Avaliação de compatibilidade
# -----------------------

print ("Carregando função para avaliação de compatibilidade de candidatos com base nos parâmetros json...")

def evaluate_candidate(text, reqs):
    txt = normalize_text(text)
    if not txt:
        logger.debug('evaluate_candidate: texto vazio')
    score = 0.0
    max_score = 0.0
    matched = []
    tokens = set(txt.split())

    for r in reqs.get('obrigatorios', []):
        r_norm = normalize_text(r)
        max_score += 2
        if re.search(r'\\b' + re.escape(r_norm) + r'\\b', txt):
            score += 2
            matched.append(r)
        else:
            parts = r_norm.split()
            if parts and all(p in tokens for p in parts):
                score += 2
                matched.append(f"{r} (tokens)")

    for r in reqs.get('diferenciais', []):
        r_norm = normalize_text(r)
        max_score += 1
        if re.search(r'\\b' + re.escape(r_norm) + r'\\b', txt):
            score += 1
            matched.append(r)
        else:
            parts = r_norm.split()
            if parts and all(p in tokens for p in parts):
                score += 1
                matched.append(f"{r} (tokens)")

    compat = round((score / max_score) * 100, 2) if max_score > 0 else 0.0
    logger.debug('evaluate_candidate: score=%s max=%s compat=%s matched=%s', score, max_score, compat, matched[:10])
    return compat, matched

# -----------------------
# Relatórios / gráficos (mantidos)
# -----------------------

def create_report_images(candidates, all_candidates=None, top_k=10):
    imgs = []
    top = candidates[:top_k]
    names = [c.get('name', f"Candidato{i}") for i,c in enumerate(top,1)]
    scores = [c['compat'] for c in top]

    if names and scores:
        plt.figure(figsize=(12,5))
        plt.barh(names[::-1], scores[::-1])
        plt.xlabel('Compatibilidade (%)')
        plt.title(f'Top {len(top)} candidatos')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        cid1 = make_msgid(domain='clickip.com.br')
        imgs.append((cid1, buf.read()))
        buf.close()

    pool = all_candidates if all_candidates is not None else candidates
    scores_all = [c['compat'] for c in pool]
    if scores_all:
        plt.figure(figsize=(12,5))
        plt.hist(scores_all, bins=10)
        plt.xlabel('Compatibilidade (%)')
        plt.ylabel('Quantidade de candidatos')
        plt.title('Distribuição de compatibilidade')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        cid2 = make_msgid(domain='clickip.com.br')
        imgs.append((cid2, buf.read()))
        buf.close()

    kw_counter = Counter()
    for c in top:
        for k in c.get('matched', []):
            k_clean = re.sub(r'\s*\(.*\)$', '', k).strip()
            kw_counter[k_clean] += 1

    if kw_counter:
        most_common = kw_counter.most_common(10)
        labels = [t[0] for t in most_common]
        values = [t[1] for t in most_common]
        plt.figure(figsize=(12,5))
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Palavras-chave mais frequentes (Top)')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        cid3 = make_msgid(domain='clickip.com.br')
        imgs.append((cid3, buf.read()))
        buf.close()

    logger.debug('create_report_images: imagens geradas=%s', len(imgs))
    return imgs


def build_html_for_candidates(top_candidates, periodo_inicio, periodo_fim, logo_cid, image_cids):
    th_html = "<th>Documento</th><th>E-mail</th><th>Função</th><th>Compatibilidade (%)</th><th>Parâmetros analisados</th>"
    tr_html = ''
    for c in top_candidates:
        tr_html += f"<tr><td>{c.get('name','-')}</td><td>{c.get('email','-')}</td><td>{c.get('vaga','-')}</td><td>{c.get('compat')}</td><td>{', '.join(c.get('matched',[]))}</td></tr>"

    charts_html = ""
    for cid in image_cids:
        charts_html += f"""<div style=\"margin-top:10px;\"><img src=\"cid:{cid[1:-1]}\" style=\"max-width:100%;height:auto;border:1px solid #ddd;margin-bottom:8px;\"></div>"""

    html = f"""
    <html><head>
      <style>
        body {{ font-family: Arial, sans-serif; color: #333; }}
        th, td {{ padding: 6px 8px; border: 1px solid #ddd; }}
        th {{ background-color: #0072C6; color: white; }}
        table {{ border-collapse: collapse; width:100%; }}
      </style>
    </head>
    <body>
      <div style="display:flex;align-items:center;">
        <div><h2>Triagem automática de CVs - Instalador de equipamentos de comunicação jr - Urucará/PA</h2><p>Período: {periodo_inicio} a {periodo_fim}</p></div>
      </div>

      <p>Top {len(top_candidates)} candidatos encontrados.</p>
      <table><tr>{th_html}</tr>{tr_html}</table>

      <h3 style="margin-top:18px;">Gráficos</h3>
      {charts_html}

      <p style="margin-top:10px;">Este é um e-mail automático.</p>
    </body></html>
    """
    return html


def save_email_eml(msg, filename_base='report_email'):
    try:
        p = Path(ATTACHMENTS_DIR) / f"{filename_base}.eml"
        with open(p, 'wb') as f:
            f.write(msg.as_bytes())
        logger.debug('E-mail salvo como EML: %s', p)
        return str(p)
    except Exception:
        logger.exception('Falha ao salvar EML')
        return None


def send_report_email(to_addrs, cc_addrs, subject, html_body, logo_path, images):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = ', '.join(to_addrs) if isinstance(to_addrs, (list,tuple)) else to_addrs
    if cc_addrs:
        msg['Cc'] = ', '.join(cc_addrs) if isinstance(cc_addrs, (list,tuple)) else cc_addrs

    msg.add_alternative(html_body, subtype='html')

    for cid, img_bytes in images:
        try:
            msg.get_payload()[0].add_related(img_bytes, maintype='image', subtype='png', cid=cid)
        except Exception:
            logger.exception('Falha ao anexar imagem ao email')

    # salvar EML para debug
    try:
        save_email_eml(msg)
    except Exception:
        logger.debug('Não foi possível salvar o eml')

    try:
        logger.info('Enviando e-mail via %s:%s (user=%s)', SMTP_HOST, SMTP_PORT, SMTP_USER)
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)
        logger.info('Relatório enviado para %s', to_addrs)
    except Exception as e:
        logger.exception('Falha ao enviar e-mail: %s', e)

# -----------------------
# DB query / Orquestração
# -----------------------

def get_checkbox_entries_between(cur, dt_from, dt_to):
    q = f"SELECT entry_id, meta_value, date_created FROM {TABLE_NAME} WHERE meta_key = %s AND date_created BETWEEN %s AND %s"
    logger.debug('Executando query: %s / params: (%s,%s)', q, dt_from, dt_to)
    cur.execute(q, ('checkbox-1', dt_from, dt_to))
    rows = cur.fetchall()
    logger.info('get_checkbox_entries_between: rows=%s', len(rows) if rows is not None else 0)
    return rows


def get_meta_value(cur, entry_id, meta_key_like):
    q = f"SELECT meta_value FROM {TABLE_NAME} WHERE entry_id = %s AND meta_key LIKE %s LIMIT 1"
    try:
        cur.execute(q, (entry_id, meta_key_like))
        r = cur.fetchone()
        logger.debug('get_meta_value entry_id=%s key=%s -> %s', entry_id, meta_key_like, (r[0] if r else None))
        return r[0] if r else None
    except Exception:
        logger.exception('Erro em get_meta_value para entry_id=%s key=%s', entry_id, meta_key_like)
        return None


def process_entries_for_vaga(cur, entries, vaga_search_text, job_reqs):
    vaga_norm = normalize_text(vaga_search_text)
    ensure_dir(ATTACHMENTS_DIR)
    processed_hashes = set()
    candidates = []

    logger.info('process_entries_for_vaga: entries=%s vaga_search_text="%s"', len(entries), vaga_search_text)

    for i, row in enumerate(entries, start=1):
        try:
            entry_id, cb_value, date_created = row
            logger.debug('Processando [%s/%s] entry_id=%s date_created=%s', i, len(entries), entry_id, date_created)
            if not cb_value:
                logger.debug('entry_id %s: checkbox vazio, pulando', entry_id)
                continue
            if vaga_norm not in normalize_text(cb_value):
                logger.debug('entry_id %s: checkbox não corresponde a vaga %s (conteudo=%s)', entry_id, vaga_search_text, str(cb_value)[:80])
                continue

            upload_meta = get_meta_value(cur, entry_id, 'upload-%')
            if not upload_meta:
                logger.warning('entry_id %s sem upload-meta, pulando', entry_id)
                continue
            file_url = parse_upload_meta(upload_meta)
            if not file_url:
                logger.warning('entry_id %s upload-meta sem URL. Conteudo preview: %s', entry_id, str(upload_meta)[:300])
                continue

            prefix = str(entry_id)
            local_file = download_file(file_url, ATTACHMENTS_DIR, prefix=prefix)
            if not local_file:
                logger.warning('Falha no download para entry_id %s url %s', entry_id, file_url)
                continue

            h = file_hash(local_file)
            if not h:
                logger.warning('Não foi possível calcular hash para %s', local_file)
                continue
            if h in processed_hashes:
                logger.info('Arquivo duplicado (hash) pulado: %s', local_file)
                try:
                    os.remove(local_file)
                except Exception:
                    pass
                continue
            processed_hashes.add(h)

            name = get_meta_value(cur, entry_id, 'name-%') or Path(local_file).name
            email_addr = get_meta_value(cur, entry_id, 'email-%') or '-'

            txt = extract_text(local_file)
            if not txt.strip():
                logger.warning('Nenhum texto extraído de %s (entry %s) - salvando amostra do arquivo para debug', local_file, entry_id)
                debug_save_binary_artifact(f"{entry_id}_raw_{Path(local_file).name}", open(local_file,'rb').read())

            # tentar mapear o requisito correto no JSON (procura por chave que contenha parte do texto da vaga)
            job_key = None
            for k in job_reqs.keys():
                if normalize_text(k).find(normalize_text(vaga_search_text)) != -1 or normalize_text(vaga_search_text) in normalize_text(k):
                    job_key = k
                    break
            if job_key is None and job_reqs:
                # fallback: pegar primeira chave
                job_key = list(job_reqs.keys())[0]

            logger.debug('job_key mapeado para entry %s: %s', entry_id, job_key)
            reqs = job_reqs.get(job_key, {}) if job_reqs else {}

            compat, matched = evaluate_candidate(txt, reqs)

            candidate = {
                'entry_id': entry_id,
                'name': name,
                'email': email_addr,
                'vaga': vaga_search_text,
                'compat': compat,
                'matched': matched,
                'local_file': local_file,
                'hash': h,
                'date_created': date_created
            }
            logger.info('Candidato processado: entry=%s compat=%s matched=%s', entry_id, compat, matched[:5])
            candidates.append(candidate)

        except Exception as e:
            logger.exception('Erro processando linha %s: %s', row, e)

    logger.info('process_entries_for_vaga: candidatos totais processados=%s', len(candidates))
    return candidates

# -----------------------
# Orquestração principal
# -----------------------

def main():
    start = datetime.now()
    logger.info('Inicio da orquestracao principal em %s', start.date())

    # carregar requisitos
    job_reqs = load_job_requirements()

    # conectar ao DB
    try:
        conn, cur = connect_db()
    except Exception:
        logger.error('Falha na conexao ao DB. Abortando.')
        return

    # periodo (últimos 7 dias por padrão)
    dt_to = datetime.now()
    dt_from = dt_to - timedelta(minutes=30)
    dt_from_s = dt_from.strftime('%Y-%m-%d %H:%M:%S')
    dt_to_s = dt_to.strftime('%Y-%m-%d %H:%M:%S')
    logger.info('Consultando entradas entre %s e %s', dt_from_s, dt_to_s)

    try:
        entries = get_checkbox_entries_between(cur, dt_from_s, dt_to_s)
    except Exception:
        logger.exception('Falha ao buscar entries do DB')
        entries = []

    # para cada vaga conhecida no JSON -> processar
    all_candidates = []
    # se job_reqs vazio irá ler todas as vagas do json
    vagas_to_check = [
    'adicione suas vagas aqui'
    ]
    if job_reqs:
        vagas_to_check = list(job_reqs.keys())
    else:
        vagas_to_check = [None]

    logger.info('Vagas a verificar: %s', vagas_to_check)

    for vaga in vagas_to_check:
        try:
            candidates = process_entries_for_vaga(cur, entries, vaga, job_reqs)
            all_candidates.extend(candidates)
        except Exception:
            logger.exception('Falha processando vaga %s', vaga)

    # ordenar por compatibilidade
    all_candidates.sort(key=lambda x: x.get('compat',0), reverse=True)
    logger.info('Total candidatos encontrados: %s', len(all_candidates))

    # criar imagens do relatorio
    images = create_report_images(all_candidates, all_candidates, top_k=TOP_K)

    # montar HTML
    periodo_inicio = dt_from_s
    periodo_fim = dt_to_s
    logo_cid = None
    image_cids = [cid for cid, _ in images]
    html = build_html_for_candidates(all_candidates[:TOP_K], periodo_inicio, periodo_fim, logo_cid, images)

    # enviar relatorio
    try:
        to_addrs = [REPORT_TO]
        cc_addrs = [REPORT_CC]
        subj = f"Relatório Triagem CVs - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        send_report_email(to_addrs, cc_addrs, subj, html, LOGO_PATH, images)
    except Exception:
        logger.exception('Erro ao enviar relatorio por e-mail')

    # finalizar
    try:
        cur.close()
        conn.close()
    except Exception:
        logger.debug('Erro ao fechar conexao com o DB (ou canal já fechado)')

    end = datetime.now()
    logger.info('Orquestração finalizada. Duração: %s segundos', (end - start).total_seconds())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception('Erro fatal no script: %s', e)
        sys.exit(1)
