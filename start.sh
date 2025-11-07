#!/bin/bash

# Script de inicio para NOESIS Prediction APIs
# Autor: OMNIA Team
# Versión: 1.0

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="noesis-prediction-api"
LOG_FILE="$SCRIPT_DIR/logs/startup.log"

# Crear directorio de logs
mkdir -p "$SCRIPT_DIR/logs"

# Funciones de utilidad
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

info() {
    log "${BLUE}[INFO]${NC} $1"
}

success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

error() {
    log "${RED}[ERROR]${NC} $1"
    exit 1
}

# Verificar dependencias
check_dependencies() {
    info "Verificando dependencias..."
    
    # Verificar Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 no está instalado"
    fi
    
    # Verificar pip
    if ! command -v pip3 &> /dev/null; then
        error "pip3 no está instalado"
    fi
    
    # Verificar Redis
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            success "Redis está disponible"
        else
            warning "Redis no responde, el sistema usará cache en memoria"
        fi
    else
        warning "Redis no está instalado, el sistema usará cache en memoria"
    fi
    
    success "Dependencias verificadas"
}

# Instalar dependencias Python
install_dependencies() {
    info "Instalando dependencias Python..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        success "Dependencias Python instaladas"
    else
        error "requirements.txt no encontrado"
    fi
}

# Configurar variables de entorno
setup_environment() {
    info "Configurando variables de entorno..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            success "Archivo .env creado desde .env.example"
            warning "Por favor, revisar y configurar las variables en .env"
        else
            warning "No se encontró .env.example, usando configuración por defecto"
        fi
    else
        success "Archivo .env existe"
    fi
}

# Cargar variables de entorno
load_environment() {
    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
        info "Variables de entorno cargadas"
    fi
}

# Crear directorios necesarios
create_directories() {
    info "Creando directorios necesarios..."
    
    directories=(
        "logs"
        "data/cache"
        "data/models"
        "data/exports"
        "webhook_logs"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        success "Directorio creado: $dir"
    done
}

# Verificar configuración
validate_config() {
    info "Validando configuración..."
    
    # Verificar puerto disponible
    PORT=${API_PORT:-8000}
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
        warning "Puerto $PORT está en uso"
    else
        info "Puerto $PORT está disponible"
    fi
    
    # Verificar variables críticas
    if [ -z "$JWT_SECRET" ]; then
        warning "JWT_SECRET no está configurado, usando valor por defecto"
    fi
    
    success "Configuración validada"
}

# Iniciar servicios
start_services() {
    info "Iniciando servicios..."
    
    # Cargar variables de entorno
    load_environment
    
    # Configurar parámetros de uvicorn
    HOST=${API_HOST:-0.0.0.0}
    PORT=${API_PORT:-8000}
    WORKERS=${API_WORKERS:-1}
    RELOAD=${API_RELOAD:-false}
    
    info "Iniciando servidor en $HOST:$PORT"
    
    # Construir comando
    CMD="uvicorn noesis_prediction_apis:app --host $HOST --port $PORT"
    
    if [ "$WORKERS" -gt "1" ]; then
        CMD="$CMD --workers $WORKERS"
    fi
    
    if [ "$RELOAD" = "true" ]; then
        CMD="$CMD --reload"
    fi
    
    if [ "$LOG_LEVEL" ]; then
        CMD="$CMD --log-level $LOG_LEVEL"
    fi
    
    info "Comando de inicio: $CMD"
    
    # Ejecutar
    exec $CMD
}

# Mostrar ayuda
show_help() {
    echo "NOESIS Prediction APIs - Script de inicio"
    echo ""
    echo "Uso: $0 [opción]"
    echo ""
    echo "Opciones:"
    echo "  start     Iniciar el servicio (por defecto)"
    echo "  dev       Iniciar en modo desarrollo"
    echo "  install   Instalar dependencias únicamente"
    echo "  check     Verificar dependencias únicamente"
    echo "  help      Mostrar esta ayuda"
    echo ""
    echo "Variables de entorno:"
    echo "  API_HOST          Host del servidor (por defecto: 0.0.0.0)"
    echo "  API_PORT          Puerto del servidor (por defecto: 8000)"
    echo "  API_WORKERS       Número de workers (por defecto: 1)"
    echo "  API_RELOAD        Modo reload (por defecto: false)"
    echo "  LOG_LEVEL         Nivel de logging (INFO, DEBUG, etc.)"
    echo ""
}

# Función principal
main() {
    case "${1:-start}" in
        "start")
            info "Iniciando NOESIS Prediction APIs..."
            check_dependencies
            install_dependencies
            setup_environment
            create_directories
            validate_config
            start_services
            ;;
        "dev")
            info "Iniciando en modo desarrollo..."
            export API_RELOAD=true
            export LOG_LEVEL=DEBUG
            check_dependencies
            install_dependencies
            setup_environment
            create_directories
            start_services
            ;;
        "install")
            check_dependencies
            install_dependencies
            ;;
        "check")
            check_dependencies
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            error "Opción desconocida: $1. Use '$0 help' para ver opciones disponibles."
            ;;
    esac
}

# Capturar Ctrl+C
trap 'error "Script interrumpido por el usuario"' INT

# Ejecutar función principal
main "$@"