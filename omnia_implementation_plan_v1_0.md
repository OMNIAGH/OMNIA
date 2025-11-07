# OMNIA MVP v1.0 - Plan de Implementación Detallado

## 🎯 Objetivo
Construir la base del sistema OMNIA con n8n, estableciendo el CORE funcional, infraestructura de datos y APIs de orquestación.

## 📅 Cronograma Estimado: 4-6 horas

### 🔧 Fase 1: Infraestructura Base (1.5-2 horas)

**Tareas Principales:**

#### 1.1 Setup de Base de Datos PostgreSQL
```sql
-- Crear tablas base
CREATE TABLE omnia_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    request_type VARCHAR(50) NOT NULL,
    priority VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    payload JSONB,
    result JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE omnia_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES omnia_sessions(id),
    module VARCHAR(50) NOT NULL,
    level VARCHAR(20) NOT NULL,
    message TEXT,
    details JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE omnia_config (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB,
    description TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE omnia_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    command JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    created_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP
);
```

#### 1.2 Configuración de Redis
```bash
# Estructura de datos en Redis
# Canales:
# - omnia_commands: LPUSH/BRPOP para comandos
# - omnia_results: PUBLISH/SUBSCRIBE para resultados
# - omnia_health: Health checks
```

#### 1.3 Setup de Credenciales
Almacenar en n8n:
- `OPENAI_API_KEY`: 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
- `ANTHROPIC_API_KEY`: e5362baf-c777-4d57-a609-6eaf1f9e87f6
- `GOOGLE_API_KEY`: AIzaSyCChjZ3ZRSKnCA2U1QKOwM4OtS54906D4g
- `SERPER_API_KEY`: e5362baf-c777-4d57-a609-6eaf1f9e87f6

### 🏗️ Fase 2: OMNIA CORE v1.0 (2-2.5 horas)

#### 2.1 Workflow Principal de Orquestación
**Nodos de N8N a crear:**

1. **Webhook Trigger** (Nodo: Webhook)
   - Endpoint: `/webhook/omnia/command`
   - Method: POST
   - Authentication: Bearer token

2. **Validar Request** (Nodo: Code)
   ```javascript
   // Validar estructura de request
   const { user_id, request_type, priority, payload } = $input.first().json;
   
   if (!user_id || !request_type) {
     return [{
       json: { error: 'Missing required fields', code: 400 }
     }];
   }
   
   return [{
     json: { 
       session_id: generateUUID(),
       user_id, 
       request_type, 
       priority: priority || 'medium',
       payload,
       status: 'pending'
     }
   }];
   ```

3. **Crear Sesión en DB** (Nodo: PostgreSQL)
   - Query: INSERT en `omnia_sessions`
   - Return: session_id

4. **Encolar Comando** (Nodo: Redis)
   - Command: LPUSH `omnia_commands`
   - Data: JSON con session_id y comando

5. **Response Handler** (Nodo: Code)
   - Generar respuesta HTTP 200
   - Session ID y status

#### 2.2 API REST Endpoints (Nodos Webhook)

**POST /api/v1/session**
```javascript
// Iniciar nueva sesión
const response = await fetch('/webhook/omnia/command', {
  method: 'POST',
  headers: { 'Authorization': 'Bearer ' + token },
  body: JSON.stringify(request_body)
});
```

**GET /api/v1/session/{session_id}**
- Query: PostgreSQL `omnia_sessions`
- Return: Status, result, timestamps

**POST /api/v1/webhook/result**
- Para recibir resultados de módulos

#### 2.3 Sistema de Logging
**Nodo: PostgreSQL - Log Writer**
```javascript
const logData = {
  session_id,
  module: 'OMNIA_CORE',
  level: 'INFO', // INFO, WARN, ERROR
  message: 'Session created',
  details: { user_id, request_type }
};
```

### 🔄 Fase 3: Procesamiento Asíncrono (1-1.5 horas)

#### 3.1 Worker Principal (Nodo: Webhook Interval)
- Poll Redis: `BRPOP omnia_commands`
- Actualizar status en DB: 'processing'
- Enrutar a API externa según request_type

#### 3.2 Procesador OpenAI (Nodo: HTTP Request)
```javascript
// Configuración para OpenAI
{
  url: 'https://api.openai.com/v1/chat/completions',
  headers: {
    'Authorization': 'Bearer {{ $credentials.openai.apiKey }}',
    'Content-Type': 'application/json'
  },
  body: {
    model: 'gpt-4',
    messages: [{ role: 'user', content: '{{ $json.payload.content }}' }],
    max_tokens: 1000
  }
}
```

#### 3.3 Result Handler (Nodo: Code)
```javascript
// Procesar resultado de API
const result = {
  session_id: $json.session_id,
  result: $json.choices[0].message.content,
  status: 'completed',
  processing_time: Date.now() - $json.start_time
};

// Actualizar DB
const updateQuery = `
  UPDATE omnia_sessions 
  SET result = $1, status = $2, updated_at = NOW()
  WHERE id = $3
`;
```

### 🧪 Fase 4: Testing y Validación (1-1.5 horas)

#### 4.1 Pruebas de API
```bash
# Test 1: Crear sesión
curl -X POST http://localhost:5678/webhook/omnia/command \
  -H "Authorization: Bearer test-token" \
  -d '{"user_id": "test", "request_type": "analysis", "priority": "high", "payload": {"content": "Hello OMNIA"}}'

# Test 2: Consultar estado
curl http://localhost:5678/webhook/omnia/session/{session_id}
```

#### 4.2 Validación de Flujos
- [ ] Creación de sesión exitosa
- [ ] Logging en PostgreSQL funcionando
- [ ] Cola Redis operativa
- [ ] OpenAI API respondiendo
- [ ] Actualización de status en DB
- [ ] Latencia <500ms

#### 4.3 Monitoreo Básico
- Dashboard en n8n con métricas
- Queries de salud de Redis y PostgreSQL
- Contador de requests por minuto

### 📊 Métricas de Éxito

#### Performance
- **Latencia**: <500ms para requests simples
- **Throughput**: >50 requests/min
- **Uptime**: >99.5%

#### Funcionalidad
- [ ] API de sesiones 100% funcional
- [ ] PostgreSQL persistiendo datos
- [ ] Redis encolando/desencolando
- [ ] OpenAI respondiendo correctamente
- [ ] Logs siendo registrados

#### Integración
- [ ] Todas las APIs conectadas
- [ ] Credenciales funcionando
- [ ] Health checks respondiendo

### 🚀 Commands de Verificación

```bash
# Verificar PostgreSQL
psql -h localhost -U postgres -c "SELECT * FROM omnia_sessions LIMIT 1;"

# Verificar Redis
redis-cli BRPOP omnia_commands 1

# Test de n8n
curl -X POST http://localhost:5678/rest/workflows/test
```

### 📁 Archivos a Generar

1. **Schema SQL**: `/workspace/db/omnia_schema.sql`
2. **Config Redis**: `/workspace/config/redis.conf`
3. **Workflow n8n**: `/workspace/n8n/omnia_core_v1_0.json`
4. **Scripts de test**: `/workspace/tests/test_omnia_core.py`
5. **Documentación API**: `/workspace/docs/api_spec_v1_0.yaml`

---

**Siguiente Paso**: Una vez completado el MVP v1.0, procederemos con el v1.1 (ENGINE CORE + ANCHOR)

**Fecha de Creación**: 2025-11-06  
**Autor**: MiniMax Agent  
**Plan Version**: v1.0
