
# NOESIS A/B Testing - Reporte de Experimento

## Información General
- **ID del Experimento**: exp_20251106_210106
- **Estado**: running
- **Fecha de Inicio**: 2025-11-06 21:01:06
- **Fecha de Finalización**: En curso
- **Early Stopping Activado**: No

## Resumen Ejecutivo

Este experimento en proceso con un lift del 0.00%. 

**Hallazgos principales:**
- El análisis confirma significancia estadística
- Se observaron mejoras en las métricas principales
- Se requiere mayor investigación

**Impacto estimado:**
- Conversiones incrementales: 43
- Ingresos adicionales: $2126.56
        

## Resultados Estadísticos
### Control
- Tasa de conversión: 0.0780
- Intervalo de confianza 95%: [0.0732, 0.0829]

### Variante
- Tasa de conversión: 0.1206
- Intervalo de confianza 95%: [0.1147, 0.1265]

### Pruebas Estadísticas
- t_test: Significativo
- chi_square: Significativo
- z_test_proportions: Significativo


## Análisis de Lift

- **Lift Absoluto**: 0.000000
- **Lift Relativo**: 0.0000
- **Lift Porcentual**: 0.00%
- **Intervalo de Confianza del Lift**: [0.0000, 0.0000]
        

## Segmentación

### new_users
- Lift: 0.0481
- Tasa de conversión: 0.1261

### returning_users
- Lift: 0.0405
- Tasa de conversión: 0.1185

### all
- Lift: 0.0425
- Tasa de conversión: 0.0993


## Recomendaciones
1. Los resultados muestran significancia estadística en: t_test, chi_square, z_test_proportions
2. Se observa un lift positivo del 54.51%. Se recomienda implementar el cambio.
3. La probabilidad bayesiana de que la variante sea mejor es muy alta (>95%).


## Apéndices

### Metodología
- Pruebas utilizadas: t-test, chi-square, análisis bayesiano
- Nivel de significancia: 0.05
- Método de early stopping: Análisis secuencial
- Análisis de lift: Bootstrap con 1000 iteraciones

### Configuración del Experimento
- Control: Layout Original
- Variantes: Layout Mejorado, Layout Minimalista
- Métrica primaria: N/A
        
        