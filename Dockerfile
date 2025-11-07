# Dockerfile para NOESIS Prediction APIs
FROM python:3.11-slim

# Crear usuario no-root
RUN useradd --create-home --shell /bin/bash noesis

# Configurar directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY noesis_prediction_apis.py .

# Cambiar permisos
RUN chown -R noesis:noesis /app

# Cambiar a usuario noesis
USER noesis

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto
CMD ["uvicorn", "noesis_prediction_apis:app", "--host", "0.0.0.0", "--port", "8000"]