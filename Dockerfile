FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema (se necessário)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivo de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY . .

# Variáveis de ambiente (podem ser sobrescritas)
ENV LLM_PROVIDER=gemini
ENV PYTHONUNBUFFERED=1

# Comando padrão
CMD ["python", "main.py"]
