# Backtest walk-forward: Kronos-small vs modelo base (BTCUSDT 1h)

Periodo: 24-oct a 7-nov 2026 (56 rondas cada 6 h, horizonte 24 velas, contexto 512, 30 trayectorias,
T=1.0, top_p=0.9, igual que el panel viral). Datos reales de Binance. Sin look-ahead.

| motor | hit-rate dirección | p-valor vs moneda | cobertura p10-p90 (ideal 80 %) | final dentro de banda | CRPS skill | MAE skill |
|---|---|---|---|---|---|---|
| kronos-small | 42,9 % (24/56) | 0,89 | 46,6 % | 41 % | -1,4 % | -31,8 % |
| bootstrap (base) | 53,6 % (30/56) | 0,34 | 73,7 % | 75 % | -3,2 % | -26,0 % |

Hallazgos:
- Kronos no acertó la dirección mejor que una moneda; quedó por debajo del modelo base.
- Sobreconfianza: su banda p10-p90 mide 2,3 % de ancho medio frente a 4,5 % del modelo base,
  y el precio real cayó dentro solo el 41 % de las veces (debería ser ~80 %).
- Sus llamadas "seguras" (P(up) >= 80 % o <= 20 %, 36 rondas) acertaron solo el 36 %.
- Sesgo alcista: 35 de 56 rondas clasificadas TRENDING UP, con 40 % de acierto.
- Ningún motor supera a "el precio no cambia" en MAE (skill negativo).

Limitaciones: 2 semanas, rondas solapadas (no independientes), una sola configuración y 30 muestras.
La prueba en vivo de 7 días (results/forward/) usa la misma configuración.
