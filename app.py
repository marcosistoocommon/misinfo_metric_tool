from threading import Lock, Thread
from uuid import uuid4

from flask import Flask, jsonify, redirect, render_template, request, url_for

from misinfo_value import main as calculate_score

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-prototype-key"

SUPPORTED_LANGS = {"en", "es"}

TEXT = {
    "en": {
        "page_title": "Bulometer",
        "prototype": "Prototype",
        "headline": "Bulometer: Misinformation value",
        "lede": "Enter the context and veracity values, and the message, to calculate its score.",
        "context_label": "Context value",
        "veracity_label": "Veracity value",
        "message_label": "Message",
        "message_placeholder": "Paste the text to analyze",
        "submit": "Analyze message",
        "language_label": "Language",
        "option_en": "English",
        "option_es": "Spanish",
        "loading_title": "Running analysis",
        "loading_subtitle": "The app is moving through each processing step.",
        "result_title": "Assigned value",
        "score_label": "Score",
        "original_message": "Original message",
        "inputs_used": "Inputs used",
        "context_line": "Context",
        "veracity_line": "Veracity",
        "another": "Analyze another message",
        "required_message": "Message is required.",
        "numeric_error": "must be a number between 0 and 1.",
        "steps": [
            "Validating inputs",
            "Normalizing language and text",
            "Analyzing patterns and tone",
            "Computing the final score",
            "Preparing the result page",
        ],
    },
    "es": {
        "page_title": "Bulómetro",
        "prototype": "Prototipo",
        "headline": "Bulómetro: Medidor de desinformación",
        "lede": "Introduce los valores de contexto y veracidad, y el mensaje, para calcular su puntuación.",
        "context_label": "Valor de contexto",
        "veracity_label": "Valor de veracidad",
        "message_label": "Mensaje",
        "message_placeholder": "Pega aquí el texto a analizar",
        "submit": "Analizar mensaje",
        "language_label": "Idioma",
        "option_en": "Inglés",
        "option_es": "Español",
        "loading_title": "Ejecutando el análisis",
        "loading_subtitle": "La aplicación está pasando por cada paso del proceso.",
        "result_title": "Valor asignado",
        "score_label": "Puntuación",
        "original_message": "Mensaje original",
        "inputs_used": "Valores usados",
        "context_line": "Contexto",
        "veracity_line": "Veracidad",
        "another": "Analizar otro mensaje",
        "required_message": "El mensaje es obligatorio.",
        "numeric_error": "debe ser un número entre 0 y 1.",
        "steps": [
            "Validando los datos",
            "Normalizando idioma y texto",
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
    "analyzing_patterns": 2,
    "computing_score": 3,
    "preparing_result": 4,
}


def update_job(token, **updates):
    with JOB_LOCK:
        job = JOB_CACHE.get(token)
        if job:
            job.update(updates)


def run_analysis_job(token, message, context_value, veracity_value, lang, redirect_url):
    def progress_callback(stage):
        update_job(token, current_step=STAGE_INDEX.get(stage, 1))

    try:
        score = calculate_score(
            input_text=message,
            context=context_value,
            veracity=veracity_value,
            progress_callback=progress_callback,
        )
        progress_callback("preparing_result")

        payload = {
            "message": message,
            "context": context_value,
            "veracity": veracity_value,
            "score": f"{score:.4f}",
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
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    lang = get_lang()
    text = ui(lang)
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    message = (request.form.get("message") or "").strip()
    context_raw = (request.form.get("context") or "").strip()
    veracity_raw = (request.form.get("veracity") or "").strip()

    errors = {}
    context_value, context_error = parse_bounded_float(context_raw, text["context_label"], lang)
    veracity_value, veracity_error = parse_bounded_float(veracity_raw, text["veracity_label"], lang)

    if not message:
        errors["message"] = text["required_message"]
    if context_error:
        errors["context"] = context_error
    if veracity_error:
        errors["veracity"] = veracity_error

    form_values = {
        "message": message,
        "context": context_raw,
        "veracity": veracity_raw,
    }

    if errors:
        if is_ajax:
            return jsonify({"ok": False, "errors": errors}), 400
        return render_template("index.html", text=text, lang=lang, form_values=form_values, errors=errors), 400

    if not is_ajax:
        score = calculate_score(
            input_text=message,
            context=context_value,
            veracity=veracity_value,
        )

        token = uuid4().hex
        RESULT_CACHE[token] = {
            "message": message,
            "context": context_value,
            "veracity": veracity_value,
            "score": f"{score:.4f}",
            "lang": lang,
        }
        return redirect(url_for("result", token=token, lang=lang))

    token = uuid4().hex
    redirect_url = url_for("result", token=token, lang=lang)

    with JOB_LOCK:
        JOB_CACHE[token] = {
            "message": message,
            "context": context_value,
            "veracity": veracity_value,
            "lang": lang,
            "current_step": 1,
            "done": False,
            "error": None,
            "redirect": redirect_url,
        }

    Thread(
        target=run_analysis_job,
        args=(token, message, context_value, veracity_value, lang, redirect_url),
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
        veracity=payload["veracity"],
        score=payload["score"],
    )


if __name__ == "__main__":
    app.run(debug=True)