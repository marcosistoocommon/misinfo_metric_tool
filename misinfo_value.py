import os
import time

os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from huggingface_hub.utils import logging as hf_hub_logging
from transformers.utils import logging as transformers_logging

hf_hub_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

from Patterns.bias import bias_value
from Patterns.violence import violence_value
from translate import translate_and_preprocess
from Patterns.propaganda import propaganda_score
from Tone.tone import tone_value
from Patterns.emotion import emotion_score, hate_speech_value
from Patterns.fallacies import fallacy_score
import argparse
from Claims.verification import false_confidence

weights = [0.25, 0.1, 0.3, 0.15, 0.2]
bias_weight, violence_weight, propaganda_weight, emotion_weight, fallacy_weight = weights

def patterns_and_tone_score(input_text, debug=False, progress_callback=None):
    """Compute combined pattern and tone scores for `input_text`.

    This runs the translation/preprocessing pipeline and all pattern
    detectors (bias, violence, propaganda, emotion, fallacies) and returns
    an aggregated pattern score together with the tone value.

    Args:
        input_text: Raw input text.
        debug: When True, also return a `details` dict with individual
            component scores.
        progress_callback: Optional callable to report progress stages.

    Returns:
        Tuple `(patterns_score, tone)` or `(patterns_score, tone, details)` when
        `debug=True`.
    """

    if progress_callback:
        progress_callback("normalizing_text")
    text = translate_and_preprocess(input_text)
    if progress_callback:
        progress_callback("analyzing_patterns")
    bias = bias_value(text) 
    violence = violence_value(text) 
    propaganda = propaganda_score(text) 
    tone = tone_value(text)
    emotion = emotion_score(text) 
    hate_speech = hate_speech_value(text) 
    fallacy = fallacy_score(text) 
    patterns_score = bias * bias_weight + (violence + hate_speech) / 2 * violence_weight + propaganda * propaganda_weight + emotion * emotion_weight + fallacy * fallacy_weight
    if debug:
        details = {
            "bias": bias,
            "violence": violence,
            "propaganda": propaganda,
            "emotion": emotion,
            "hate_speech": hate_speech,
            "fallacy": fallacy,
            "tone": tone,
        }
        return patterns_score, tone, details
    return patterns_score, tone

def analyze_message(input_text, context, debug=False, progress_callback=None):
    """Analyze a message and compute the overall misinformation payload.

    Runs pattern/tone analysis, translation, and content verification to
    produce a single `payload` dictionary containing the overall score
    and component values. When `debug=True` the `details` dict is included.
    """
    input_text = translate_and_preprocess(input_text)
    if debug:
        patterns_score, tone_score, details = patterns_and_tone_score(input_text, debug=True, progress_callback=progress_callback)
    else:
        patterns_score, tone_score = patterns_and_tone_score(input_text, progress_callback=progress_callback)
        details = None
    verification_value = false_confidence(input_text, progress_callback=progress_callback)

    if progress_callback:
        progress_callback("computing_score")

    context_value = float(context)
    score = (patterns_score * 0.4) + (tone_score * 0.2) + (context_value * 0.1) + (verification_value * 0.3)

    payload = {
        "score": score,
        "context": context_value,
        "verification": verification_value,
        "patterns": patterns_score,
        "tone": tone_score,
    }

    if progress_callback:
        progress_callback("preparing_result")

    if debug:
        payload["details"] = details or {}
    return payload


def calculate_misinformation_score(input_text, context, veracity=None, debug=False, progress_callback=None):
    """Convenience wrapper returning only the score (or score+details).

    When called programmatically with `veracity` provided the function
    returns immediately, otherwise it acts as the library entrypoint used
    by the CLI below.
    """

    result = analyze_message(input_text, context, debug=debug, progress_callback=progress_callback)
    if debug:
        return result["score"], {**(result.get("details") or {}), "context": result["context"], "verification": result["verification"], "patterns": result["patterns"], "tone": result["tone"]}
    return result["score"]


def main(input_text=None, context=None, veracity=None, debug=False, progress_callback=None):
    """CLI entrypoint for computing misinformation scores.

    When invoked without CLI args the function prompts the user for input.
    When called programmatically with `input_text`, `context` and `veracity`
    provided it returns the computed score or (score, details) when
    `debug=True`.
    """

    if input_text is not None and context is not None and veracity is not None:
        return calculate_misinformation_score(
            input_text,
            context,
            veracity,
            debug=debug,
            progress_callback=progress_callback,
        )

    parser = argparse.ArgumentParser(description="Calculate misinformation score")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show individual pattern and tone component values",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test cases for patterns and tone components",
    )
    args = parser.parse_args()

    if args.test:
        print(weights)
        starttime= time.time()
        data = [
        # --- DESINFORMACIÓN (25, mentira=1) ---
        (0.82, 1, "Cada vez más expertos cuestionan las cifras oficiales, pero los medios no lo cuentan"),
        (0.65, 1, "No es casualidad que los precios suban justo después de ciertas decisiones políticas"),
        (0.61, 1, "Muchos estudios están financiados por grandes empresas, no son tan independientes"),
        (0.77, 1, "Cada vez hay más censura en redes cuando alguien dice algo incómodo"),
        (0.54, 1, "Hay países que están retirando medidas que aquí siguen imponiendo"),
        (0.43, 1, "Los datos no cuadran con lo que vemos en la vida diaria"),
        (0.68, 1, "Muchos profesionales discrepan pero no pueden hablar abiertamente"),
        (0.66, 1, "Antes no había tantos problemas hasta que cambiaron ciertas normas"),
        (0.59, 1, "Curioso que solo se escuche una versión del tema"),
        (0.84, 1, "Hay demasiados intereses económicos detrás de esto"),
        (0.72, 1, "Cada vez más gente investiga por su cuenta y llega a otras conclusiones"),
        (0.48, 1, "Los medios tradicionales están perdiendo credibilidad por algo"),
        (0.63, 1, "No muestran todos los datos, solo los que les interesan"),
        (0.69, 1, "Muchos expertos independientes están siendo ignorados"),
        (0.81, 1, "Algo no encaja con las cifras oficiales"),
        (0.87, 1, "Cada vez hay más testimonios que contradicen la versión oficial"),
        (0.65, 1, "No todo es como lo pintan en televisión"),
        (0.66, 1, "Hay demasiadas coincidencias para que sea casualidad"),
        (0.52, 1, "Los datos cambian constantemente, eso no genera confianza"),
        (0.49, 1, "Muchos estudios contradictorios no llegan a publicarse"),
        (0.74, 1, "Cada vez más gente duda de lo que dicen las autoridades"),
        (0.63, 1, "Es raro que no se debata este tema abiertamente"),
        (0.58, 1, "Hay demasiada presión para que todos piensen igual"),
        (0.81, 1, "Los expertos que discrepan desaparecen del debate"),
        (0.69, 1, "No están contando toda la historia"),

        # --- ODIO (25, mentira=0.75) ---
        (0.45, 0.75, "Siempre son los mismos los que causan problemas en este país"),
        (0.67, 0.75, "Ese grupo nunca aporta nada positivo"),
        (0.38, 0.75, "Estoy harto de ver a esa gente en todas partes"),
        (0.72, 0.75, "No deberían tener tanto protagonismo"),
        (0.26, 0.75, "Esa comunidad solo genera conflictos"),
        (0.51, 0.75, "No quiero compartir espacio con personas así"),
        (0.34, 0.75, "Siempre perjudican al resto"),
        (0.64, 0.75, "No aportan nada útil a la sociedad"),
        (0.29, 0.75, "Deberían dejar de imponer su forma de ver las cosas"),
        (0.57, 0.75, "Es imposible confiar en gente así"),
        (0.43, 0.75, "Siempre buscan aprovecharse del sistema"),
        (0.75, 0.75, "No deberían tener las mismas oportunidades"),
        (0.37, 0.75, "Ese tipo de personas solo trae problemas"),
        (0.62, 0.75, "No encajan en una sociedad que funcione"),
        (0.48, 0.75, "Siempre están generando polémica"),
        (0.21, 0.75, "Nunca hacen nada constructivo"),
        (0.53, 0.75, "Es mejor mantener distancia con ellos"),
        (0.68, 0.75, "No deberían participar en decisiones importantes"),
        (0.32, 0.75, "Solo complican las cosas"),
        (0.59, 0.75, "No aportan ningún valor real"),
        (0.27, 0.75, "Siempre crean división"),
        (0.71, 0.75, "No deberían tener tanta visibilidad"),
        (0.46, 0.75, "Es mejor no confiar en ese grupo"),
        (0.35, 0.75, "Nunca ayudan a mejorar la situación"),
        (0.63, 0.75, "Son parte del problema, no de la solución"),

        # --- INFORMATIVO (25, mentira=0) ---
        (0.12, 0, "El gobierno presenta un nuevo plan económico para el próximo año"),
        (0.08, 0, "Un estudio analiza el impacto del teletrabajo en la productividad"),
        (0.14, 0, "La inflación se mantiene estable según los últimos datos"),
        (0.05, 0, "Expertos destacan el aumento del uso de energías renovables"),
        (0.09, 0, "Se inaugura un nuevo centro de investigación en la ciudad"),
        (0.03, 0, "El paro juvenil desciende ligeramente este trimestre"),
        (0.11, 0, "Un informe analiza el acceso a la vivienda en grandes ciudades"),
        (0.07, 0, "La educación digital continúa expandiéndose en escuelas"),
        (0.15, 0, "Se aprueban nuevas medidas para mejorar el transporte público"),
        (0.04, 0, "El turismo muestra signos de recuperación este año"),
        (0.10, 0, "Investigadores presentan avances en medicina preventiva"),
        (0.06, 0, "El consumo energético se reduce en los últimos meses"),
        (0.13, 0, "Se publican nuevos datos sobre crecimiento económico"),
        (0.02, 0, "El sistema sanitario incorpora nuevas tecnologías"),
        (0.14, 0, "Un estudio evalúa el impacto del cambio climático"),
        (0.05, 0, "Las exportaciones aumentan según el último informe"),
        (0.09, 0, "Se anuncian mejoras en infraestructuras urbanas"),
        (0.12, 0, "La población activa crece ligeramente"),
        (0.03, 0, "Se amplían los horarios del transporte público"),
        (0.11, 0, "Un informe analiza hábitos de consumo actuales"),
        (0.07, 0, "Se incrementa la inversión en educación"),
        (0.02, 0, "Expertos analizan tendencias tecnológicas"),
        (0.10, 0, "Se publican datos sobre empleo en el sector servicios"),
        (0.06, 0, "El mercado laboral muestra signos de estabilidad"),
        (0.15, 0, "Nuevas políticas buscan mejorar la sostenibilidad"),

        # --- OPINIÓN (25, mentira=0.33) ---
        (0.22, 0.33, "Creo que el teletrabajo debería mantenerse a largo plazo"),
        (0.35, 0.33, "En mi opinión, la educación necesita cambios profundos"),
        (0.18, 0.33, "Pienso que la tecnología está cambiando nuestras relaciones"),
        (0.39, 0.33, "Para mí, vivir en ciudades pequeñas es mejor"),
        (0.16, 0.33, "Siento que deberíamos consumir de forma más responsable"),
        (0.27, 0.33, "Creo que el deporte es clave para el bienestar mental"),
        (0.12, 0.33, "En mi opinión, las redes sociales tienen demasiado peso"),
        (0.31, 0.33, "Pienso que la música influye mucho en nuestro ánimo"),
        (0.20, 0.33, "Creo que viajar abre mucho la mente"),
        (0.28, 0.33, "Personalmente, prefiero trabajar en equipo"),
        (0.24, 0.33, "Opino que deberíamos desconectar más de la tecnología"),
        (0.37, 0.33, "Creo que la lectura sigue siendo fundamental"),
        (0.15, 0.33, "Para mí, la alimentación saludable es clave"),
        (0.33, 0.33, "Pienso que la educación emocional debería ser obligatoria"),
        (0.19, 0.33, "Creo que el equilibrio entre vida personal y trabajo es esencial"),
        (0.36, 0.33, "En mi opinión, el transporte público debería mejorar"),
        (0.21, 0.33, "Pienso que la sociedad cambia demasiado rápido"),
        (0.38, 0.33, "Creo que el descanso es tan importante como trabajar"),
        (0.26, 0.33, "Personalmente, valoro más el tiempo libre que el dinero"),
        (0.30, 0.33, "Opino que deberíamos cuidar más el medio ambiente"),
        (0.14, 0.33, "Creo que aprender idiomas es muy importante"),
        (0.34, 0.33, "Pienso que la creatividad está infravalorada"),
        (0.23, 0.33, "En mi opinión, el sistema educativo necesita innovar"),
        (0.17, 0.33, "Creo que la tecnología debería usarse con moderación"),
        (0.32, 0.33, "Para mí, la salud mental debería ser prioridad"),
        ]
        test_scores = []
        acc_yes = 0
        acc_no = 0
        for i, ind_test in enumerate(data):
            context = ind_test[0]
            text = ind_test[2]
            misinfo_score = analyze_message(text, context)["score"]
            test_scores.append(misinfo_score)
             # Progress bar
            progress = (i + 1) / len(data)
            bar_length = 50
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f'\rProgress: {i+1}/{len(data)} [{bar}]', end='', flush=True)
        print()  # New line after progress bar completes
        for i in range(len(test_scores)):
            if i<50 and test_scores[i] >= 0.55:
                acc_yes += 1
            elif i>=50 and test_scores[i] <= 0.2:
                acc_no += 1
        stoptime = time.time()
        print(f"Test completed in {stoptime-starttime:.2f} seconds")
        print(f"Positive accuracy: {acc_yes/50:.2f}")
        print(f"Negative accuracy: {acc_no/50:.2f}")
        print(f"Overall accuracy: {(acc_yes+acc_no)/100:.2f}")
        return

    input_text = input("Enter the text to analyze for misinformation: ")
    context = input("Enter the value of the context: ")
    veracity = input("Enter the value of the veracity: ")

    result = calculate_misinformation_score(input_text, context, veracity, debug=args.debug)

    if args.debug:
        misinfo_score, details = result
        print("Pattern and tone component values:")
        for key, value in details.items():
            print(f"- {key}: {value:.4f}")
    else:
        misinfo_score = result
    print(f"Misinformation Score: {misinfo_score:.4f}")




if __name__ == "__main__":
    main()

