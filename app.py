from threading import Lock, Thread
from uuid import uuid4

from flask import Flask, jsonify, redirect, render_template, request, url_for

from misinfo_value import analyze_message as calculate_analysis
from Claims.verification import ClaimeAIError, agent_error, initialize_agent

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-prototype-key"

CLAIMEAI_ERROR = None

try:
    initialize_agent()
except ClaimeAIError as exc:
    CLAIMEAI_ERROR = str(exc)
except Exception as exc:
    CLAIMEAI_ERROR = str(exc)

SUPPORTED_LANGS = {"en", "es"}

TEXT = {
    "en": {
        "page_title": "Bulometer",
        "prototype": "Prototype",
        "headline": "Bulometer: Misinformation value",
        "lede": "Enter the context value and the message. Verification now runs in the background.",
        "context_label": "Context value",
        "message_label": "Message",
        "message_placeholder": "Paste the text to analyze",
        "submit": "Analyze message",
        "language_label": "Language",
        "option_en": "English",
        "option_es": "Spanish",
        "loading_title": "Running analysis",
        "loading_subtitle": "The app is extracting claims, verifying them, and scoring the text.",
        "result_title": "Assigned value",
        "score_label": "Score",
        "original_message": "Original message",
        "inputs_used": "Component values",
        "context_line": "Context",
        "verification_line": "Verification",
        "patterns_line": "Patterns",
        "tone_line": "Tone",
        "another": "Analyze another message",
        "required_message": "Message is required.",
        "backend_error_label": "Verification backend error",
        "numeric_error": "must be a number between 0 and 1.",
        "steps": [
            "Validating inputs",
            "Extracting claims",
            "Verifying claims",
            "Analyzing patterns and tone",
            "Computing the final score",
            "Preparing the result page",
        ],
    },
    "es": {
        "page_title": "Bulómetro",
        "prototype": "Prototipo",
        "headline": "Bulómetro: Medidor de desinformación",
        "lede": "Introduce el valor de contexto y el mensaje. La verificación se ejecuta en segundo plano.",
        "context_label": "Valor de contexto",
        "message_label": "Mensaje",
        "message_placeholder": "Pega aquí el texto a analizar",
        "submit": "Analizar mensaje",
        "language_label": "Idioma",
        "option_en": "Inglés",
        "option_es": "Español",
        "loading_title": "Ejecutando el análisis",
        "loading_subtitle": "La aplicación está extrayendo claims, verificándolos y calculando la puntuación.",
        "result_title": "Valor asignado",
        "score_label": "Puntuación",
        "original_message": "Mensaje original",
        "inputs_used": "Valores de los componentes",
        "context_line": "Contexto",
        "verification_line": "Verificación",
        "patterns_line": "Patrones",
        "tone_line": "Tono",
        "another": "Analizar otro mensaje",
        "required_message": "El mensaje es obligatorio.",
        "backend_error_label": "Error del backend de verificación",
        "numeric_error": "debe ser un número entre 0 y 1.",
        "steps": [
            "Validando los datos",
            "Extrayendo claims",
            "Verificando claims",
            "Analizando patrones y tono",
            "Calculando la puntuación final",
            "Preparando la página de resultados",
        ],
    },
}

RESULT_CACHE = {}
JOB_CACHE = {}
JOB_LOCK = Lock()
STAGE_INDEX = {
    "normalizing_text": 1,
    "extracting_claims": 1,
    "verifying_claims": 2,
    "analyzing_patterns": 3,
    "computing_score": 4,
    "preparing_result": 5,
    "aggregating_verdict": 4,
}


def update_job(token, **updates):
    with JOB_LOCK:
        job = JOB_CACHE.get(token)
        if job:
            job.update(updates)


def run_analysis_job(token, message, context_value, lang, redirect_url):
    def progress_callback(stage):
        update_job(token, current_step=STAGE_INDEX.get(stage, 1))

    try:
        result = calculate_analysis(
            input_text=message,
            context=context_value,
            progress_callback=progress_callback,
        )

        payload = {
            "message": message,
            "context": result["context"],
            "verification": result["verification"],
            "patterns": result["patterns"],
            "tone": result["tone"],
            "score": f"{result['score']:.4f}",
            "lang": lang,
        }

        with JOB_LOCK:
            RESULT_CACHE[token] = payload
            if token in JOB_CACHE:
                JOB_CACHE[token].update(
                    {
                        "done": True,
                        "current_step": len(TEXT[lang]["steps"]),
                        "redirect": redirect_url,
                        "result": payload,
                    }
                )
    except Exception as exc:
        update_job(token, done=True, error=str(exc))


def get_lang(value=None):
    candidate = value or request.values.get("lang") or request.args.get("lang") or "en"
    return candidate if candidate in SUPPORTED_LANGS else "en"


def ui(lang):
    return TEXT[get_lang(lang)]


def parse_bounded_float(raw_value, field_name, lang):
    message = ui(lang)["numeric_error"]
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None, f"{field_name} {message}"

    if value < 0 or value > 1:
        return None, f"{field_name} {message}"

    return value, None


@app.route("/", methods=["GET"])
def index():
    lang = get_lang()
    text = ui(lang)
    return render_template(
        "index.html",
        text=text,
        lang=lang,
        form_values={},
        errors={},
        backend_error=CLAIMEAI_ERROR,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    lang = get_lang()
    text = ui(lang)
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    message = (request.form.get("message") or "").strip()
    context_raw = (request.form.get("context") or "").strip()

    errors = {}
    context_value, context_error = parse_bounded_float(context_raw, text["context_label"], lang)

    if not message:
        errors["message"] = text["required_message"]
    if context_error:
        errors["context"] = context_error

    form_values = {
        "message": message,
        "context": context_raw,
    }

    if CLAIMEAI_ERROR:
        errors["backend"] = CLAIMEAI_ERROR

    if errors:
        if is_ajax:
            return jsonify({"ok": False, "errors": errors}), 400
        return render_template("index.html", text=text, lang=lang, form_values=form_values, errors=errors, backend_error=CLAIMEAI_ERROR), 400

    if not is_ajax:
        result = calculate_analysis(input_text=message, context=context_value)

        token = uuid4().hex
        RESULT_CACHE[token] = {
            "message": message,
            "context": result["context"],
            "verification": result["verification"],
            "patterns": result["patterns"],
            "tone": result["tone"],
            "score": f"{result['score']:.4f}",
            "lang": lang,
        }
        return redirect(url_for("result", token=token, lang=lang))

    token = uuid4().hex
    redirect_url = url_for("result", token=token, lang=lang)

    with JOB_LOCK:
        JOB_CACHE[token] = {
            "message": message,
            "context": context_value,
            "lang": lang,
            "current_step": 1,
            "done": False,
            "error": None,
            "redirect": redirect_url,
        }

    Thread(
        target=run_analysis_job,
        args=(token, message, context_value, lang, redirect_url),
        daemon=True,
    ).start()

    return jsonify(
        {
            "ok": True,
            "token": token,
            "status_url": url_for("progress", token=token, lang=lang),
            "redirect": redirect_url,
            "current_step": 1,
            "total_steps": len(text["steps"]),
        }
    )


@app.route("/progress/<token>", methods=["GET"])
def progress(token):
    with JOB_LOCK:
        job = JOB_CACHE.get(token)

    if not job:
        return jsonify({"ok": False, "error": "Job not found."}), 404

    response = {
        "ok": True,
        "current_step": job["current_step"],
        "total_steps": len(TEXT[job["lang"]]["steps"]),
        "done": job["done"],
        "error": job["error"],
        "lang": job["lang"],
    }

    if job["done"]:
        response["redirect"] = job["redirect"]

    return jsonify(response)


@app.route("/result/<token>", methods=["GET"])
def result(token):
    requested_lang = request.args.get("lang")
    lang = requested_lang if requested_lang in SUPPORTED_LANGS else None
    payload = RESULT_CACHE.get(token)

    if not payload:
        return redirect(url_for("index", lang=lang or "en"))

    if lang is None and payload.get("lang") in SUPPORTED_LANGS:
        lang = payload["lang"]
    if lang is None:
        lang = "en"

    text = ui(lang)
    return render_template(
        "result.html",
        text=text,
        lang=lang,
        message=payload["message"],
        context=payload["context"],
        verification=payload["verification"],
        patterns=payload["patterns"],
        tone=payload["tone"],
        score=payload["score"],
        backend_error=CLAIMEAI_ERROR,
    )


if __name__ == "__main__":
    app.run(debug=True)