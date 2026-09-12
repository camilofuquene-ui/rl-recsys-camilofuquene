"""Reto de la sesion 6 · Gane a dos lineas de codigo.

═══════════════════════════════════════════════════════════════════════════
QUE PASA
═══════════════════════════════════════════════════════════════════════════

Sobre los datos de ASSISTments, con particion TEMPORAL, esto es lo que sale:

    recomendador                 Recall@10   nDCG@10   cobertura
    repetir lo propio               0,7009    0,5880       0,973
    popularidad                     0,4121    0,2738       0,090
    kNN por items                   0,2613    0,1857       0,856
    al azar                         0,0990    0,0683       1,000
    novedad pura                    0,0949    0,0766       0,649

Lea la primera fila y la tercera. **El filtro colaborativo pierde contra una
linea base de dos lineas**, y por casi tres veces. Y «repetir lo propio» es
literalmente esto:

    los items que este usuario ya practico, el mas repetido primero,
    y despues los mas populares para rellenar

Lea tambien la ultima fila. «Novedad pura», que solo recomienda cosas que el
usuario NO ha hecho, saca menos que recomendar AL AZAR. La intuicion de que un
recomendador no debe repetir es, en estos datos, el error mas caro posible.

═══════════════════════════════════════════════════════════════════════════
QUE HAY QUE HACER
═══════════════════════════════════════════════════════════════════════════

Dos cosas, y la primera se entrega aunque la segunda no salga.

1. **El diagnostico, escrito en su bitacora antes de tocar el codigo.** Por que
   gana repetir. Que dice eso sobre estos datos. Y una prediccion: cuanto cree
   que va a sacar su recomendador.

2. **El recomendador.** Rellene ``mi_recomendador`` para superar a «repetir lo
   propio» sin encoger el catalogo. Tiene todo ``rlrs.recomendacion``
   disponible y puede escribir el suyo desde cero.

    uv run python scripts/reto6.py        mide y da el veredicto
    uv run pytest tests/test_reto6.py     comprueba el contrato

═══════════════════════════════════════════════════════════════════════════

Aviso: hay una forma facil de subir el Recall que consiste en recomendarle a
todo el mundo lo mismo. Por eso hay un criterio de cobertura, y por eso el
arnes le ensena las cuatro cifras y no una.
"""

from __future__ import annotations

import numpy as np

from rlrs.recomendacion import (  # noqa: F401
    Particion,
    factorizacion_implicita,
    knn_items,
    por_popularidad,
    repetir_lo_propio,
    _con_nombre

)

"""MI PREDICCIÓN
    
Repetir lo propiO gana porque ordena lo que el usuario ya practicó por frecuencia total — 
cuántas veces lo hizo en toda su historia. Mi sospecha es que eso no es lo más preciso: 
en una partición temporal, lo que mejor predice qué va a hacer el usuario justo después 
del corte no es "qué tanto lo practicó en general", sino qué fue lo último que practicó — 
porque los tutores suelen insistir varias veces seguidas en la misma habilidad dentro de una sesión.

Voy a reordenar esos mismos ítems propios, lo más reciente primero en vez de por frecuencia. 
Espero una mejora en Recall@10, porque solo estoy cambiando el criterio de orden, no añadiendo información nueva."""


def mi_recomendador(particion: Particion):
    n = particion.n_items

    cuenta = np.zeros(n)
    for tr in particion.para_ajustar:
        np.add.at(cuenta, tr, 1)
    respaldo = [int(i) for i in np.argsort(cuenta)[::-1]]

    # Matriz de similitud item-item por coseno, igual que knn_items.
    R = np.zeros((len(particion.para_ajustar), n), dtype=float)
    for u, tr in enumerate(particion.para_ajustar):
        R[u, np.unique(tr)] = 1.0
    normas = np.linalg.norm(R, axis=0)
    normas[normas == 0] = 1.0
    S = (R.T @ R) / np.outer(normas, normas)
    np.fill_diagonal(S, 0.0)

    vecinos = 20
    if vecinos < n:
        umbral = np.partition(S, -vecinos, axis=1)[:, -vecinos][:, None]
        S = np.where(S >= umbral, S, 0.0)

    def recomendar(historial):
        propios = np.bincount(historial, minlength=n).astype(float)

        perfil = np.zeros(n)
        np.add.at(perfil, historial, 1.0)
        afinidad = S.T @ perfil
        if afinidad.max() > 0:
            afinidad = afinidad / afinidad.max()   

        puntaje = propios * 1000.0 + afinidad

        orden = [int(i) for i in np.argsort(puntaje)[::-1]]
        vistos = set(orden[:50])
        resto = [i for i in respaldo if i not in vistos]
        return (orden[:50] + resto)[:50]
    
    return _con_nombre(recomendar, "propio + afinidad kNN")

    """Devuelve una funcion que recomienda. **Esto es lo que usted escribe.**

    Parameters
    ----------
    particion:
        Tiene ``train`` (la lista de historiales que SI se pueden mirar),
        ``n_items`` y ``nombre``. **No tiene el conjunto de prueba**, y esa
        ausencia es lo unico que impide hacer trampa sin querer.

    Returns
    -------
    callable
        Recibe el historial de un usuario, como arreglo de items, y devuelve
        una lista de items ordenada de mejor a peor.
    """
    # ── su respuesta va aqui ──────────────────────────────────────────────
    #return knn_items(particion, vecinos=20)
