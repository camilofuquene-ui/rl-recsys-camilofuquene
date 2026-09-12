"""Reto de la sesion 5 · La politica que se queda clavada.

═══════════════════════════════════════════════════════════════════════════
QUE PASA
═══════════════════════════════════════════════════════════════════════════

Meridiano visto como bandido: veinte brazos, uno por habilidad, y en cada
ronda se elige uno. La politica avida pura hace lo obvio, que es quedarse con
el brazo que mejor le ha ido. Y le pasa esto:

    rondas    avida pura      UCB1      quien gana
        25       +0,0118    +0,0107      empatan
       100       +0,0160    +0,0393      UCB1, y por el doble
       500       +0,0160    +0,1933      UCB1, y por doce veces
      1500       +0,0160    +0,2092      UCB1, y por trece

Lea la columna de la avida despacio. **De la ronda 100 a la 1500 no gana ni
una milesima.** Mil cuatrocientas rondas mas y el mismo numero, hasta el cuarto
decimal. Se quedo clavada, y toca 2,3 brazos de veinte.

Y fijese en la primera fila: con veinticinco rondas **empata** con UCB1. No es
una politica tonta, es una politica con prisa. El problema aparece despues.

Lo que le pasa es que en cuanto un brazo le da algo, deja de mirar los demas, y
ese brazo **se agota**, porque el estudiante ya domina esa habilidad y
practicarla otra vez ya no le ensena nada. La avida se queda cobrando de un
pozo seco.

═══════════════════════════════════════════════════════════════════════════
QUE HAY QUE HACER
═══════════════════════════════════════════════════════════════════════════

Dos cosas, y la primera se entrega aunque la segunda no salga.

1. **El diagnostico, escrito en tu bitacora antes de tocar el codigo.** Por
   que se queda clavada. Que le pasa al brazo elegido a medida que se tira de
   el. Y una prediccion: cuanto crees que va a ganar tu arreglo.

2. **El arreglo.** Rellena ``mi_politica`` con una politica de bandido que
   reparta mejor las tiradas. Tienes ``epsilon_avida``, ``ucb1``, ``thompson``
   y ``linucb`` en ``rlrs.bandidos``, y puedes escribir la tuya desde cero si
   prefieres.

    uv run python scripts/reto5.py        entrena y mide, e imprime el veredicto
    uv run pytest tests/test_reto5.py     comprueba el contrato

═══════════════════════════════════════════════════════════════════════════

Aviso, y va en serio: hay una linea base tonta que va a costar mucho superar,
y descubrir cual es vale mas que superarla. Cuando la vea, anotela en la
bitacora antes de intentar ganarle.


#Creo que la política ávida pura se queda clavada porque, apenas un brazo le da una buena recompensa 
# inicial, deja de explorar los demás — y como los brazos aquí se "agotan" con el uso (simulan a un 
# estudiante que ya domina esa habilidad), ese primer brazo bueno termina siendo un pozo seco al que 
# la política sigue volviendo para siempre.
#Voy a probar UCB1, que agrega un bono de exploración a los brazos poco tocados, sin importar qué 
# tan bien le haya ido al brazo actual. Espero que gane alrededor de +0,18 a +0,20 a 500 rondas 
# (muy por encima del +0,1400 pedido) y que toque los 20 de 20 brazos, porque UCB1 obliga a probar 
# cada brazo al menos una vez al inicio.
# #Criterio de abandono: si no supera +0,1400 o toca menos de 15 brazos, pruebo con Thompson o LinUCB."""


from __future__ import annotations

from rlrs.bandidos import (  # noqa: F401
    Corrida,
    epsilon_avida,
    linucb,
    thompson,
    ucb1,
)

    
def mi_politica(bandido, rondas: int, seed: int) -> Corrida:
    """Elige un brazo en cada ronda. **Esto es lo que usted escribe.**

    Parameters
    ----------
    bandido:
        Tiene ``n_brazos`` y ``tirar(brazo, rng) -> recompensa``. Nada mas. No
        hay estado que consultar: esa es toda la gracia de un bandido.
    rondas:
        Cuantas veces se elige. Hay que hacer exactamente esas.
    seed:
        La semilla. Dos llamadas con la misma semilla tienen que dar lo mismo.

    Returns
    -------
    Corrida
        Lo que devuelven las politicas de ``rlrs.bandidos``.
    """
    # ── su respuesta va aqui ──────────────────────────────────────────────
    #return ucb1(bandido, rondas, seed=seed)
    #return linucb(bandido, rondas, seed=seed)
    #return thompson(bandido, rondas, seed=seed)
    return epsilon_avida(bandido, rondas, epsilon=0.6, seed=seed)

"""
UCB1 dio +0,1936 a 500 rondas, sobre el umbral de +0,1400, tocando los 20/20 brazos — la predicción se cumplió casi exacto.

Probé las otras tres opciones para comparar:

LinUCB mejoró aún más que UCB1: su incertidumbre se calcula con una regresión ajustada a los datos 
reales de cada brazo (una "elipse de confianza"), más precisa que la fórmula genérica de UCB1 — 
por eso explora de forma más eficiente.
Thompson no sirvió: exige recompensas de tipo éxito/fracaso (0 o 1), y este reto usa dominio 
(cuánto aprende el estudiante), que no encaja en ese supuesto. Es una limitación de diseño del método, no un error mío.
ε-ávida con ε=0,1 no alcanzó el umbral: solo explora al azar el 10% de las rondas, insuficiente 
para escapar del brazo agotado. Con ε=0,2 sí lo superó, pero nunca de forma clara: 
como la explotación (80% del tiempo) sigue cayendo sobre el brazo agotado, esta política 
se acerca a random en vez de superarlo con margen — a diferencia de UCB1/LinUCB, que exploran 
dirigidamente hacia lo menos tocado y sí le ganan a random con claridad.

Conclusión: el resultado no contradice la predicción, y además mostró algo extra: no basta con 
"explorar más" (ε alto) para resolver el problema — hace falta explorar dirigidamente hacia 
lo que no se ha tocado, que es justo lo que distingue a UCB1/LinUCB de una ávida con ruido."""