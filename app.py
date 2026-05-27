import os
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for

from misinfo_value import analyze_message as calculate_analysis
from Context.context import analyze_x_url
from Claims.verification import ClaimeAIError, initialize_agent

load_dotenv(Path(__file__).resolve().parent / ".env")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ["SECRET_KEY"]

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
        "lede": "Paste an X post URL and the app will extract the post text, compute the context value, and score the message.",
        "url_label": "X post URL",
        "url_placeholder": "https://x.com/username/status/1234567890",
        "submit": "Analyze X post",
        "language_label": "Language",
        "option_en": "English",
        "option_es": "Spanish",
        "loading_title": "Running analysis",
        "loading_subtitle": "The app is extracting the post text, evaluating the context, and computing the score.",
        "result_title": "Assigned value",
        "score_label": "Score",
        "original_message": "Extracted message",
        "inputs_used": "Values used",
        "url_line": "X URL",
        "context_line": "Context",
        "verification_line": "Fakeness",
        "patterns_line": "Patterns",
        "tone_line": "Tone",
        "another": "Analyze another X post",
        "required_url": "X URL is required.",
        "url_error": "Please enter a valid X status URL.",
        "backend_error_label": "Verification backend error",
        "numeric_error": "must be a number between 0 and 1.",
        "steps": [
            "Extracting tweet text and context",
            "Analyzing patterns and tone",
            "Extracting claims",
            "Verifying claims",
            "Computing the final score",
            "Preparing the result page",
        ],
    },
    "es": {
        "page_title": "Bulómetro",
        "prototype": "Prototipo",
        "headline": "Bulómetro: Medidor de desinformación",
        "lede": "Pega una URL de X y la aplicación extraerá el texto, calculará el contexto y puntuará el mensaje.",
        "url_label": "URL del post de X",
        "url_placeholder": "https://x.com/usuario/status/1234567890",
        "submit": "Analizar post de X",
        "language_label": "Idioma",
        "option_en": "Inglés",
        "option_es": "Español",
        "loading_title": "Ejecutando el análisis",
        "loading_subtitle": "La aplicación está extrayendo el texto del post, evaluando el contexto y calculando la puntuación.",
        "result_title": "Valor asignado",
        "score_label": "Puntuación",
        "original_message": "Mensaje extraído",
        "inputs_used": "Valores utilizados",
        "url_line": "URL de X",
        "context_line": "Contexto",
        "verification_line": "Falsedad",
        "patterns_line": "Patrones",
        "tone_line": "Tono",
        "another": "Analizar otro post de X",
        "required_url": "La URL de X es obligatoria.",
        "url_error": "Introduce una URL de X válida.",
        "backend_error_label": "Error del backend de verificación",
        "numeric_error": "debe ser un número entre 0 y 1.",
        "steps": [
            "Extrayendo el texto y el contexto",
            "Analizando patrones y tono",
            "Extrayendo claims",
            "Verificando claims",
            "Calculando la puntuación final",
            "Preparando la página de resultados",
        ],
    },
}

RESULT_CACHE = {}
JOB_CACHE = {}
JOB_LOCK = Lock()
STAGE_INDEX = {
    "extracting_context": 0,
    "normalizing_text": 1,
    "analyzing_patterns": 1,
    "extracting_claims": 2,
    "verifying_claims": 3,
    "aggregating_verdict": 3,
    "computing_score": 4,
    "preparing_result": 5,
}


def update_job(token, **updates):
    with JOB_LOCK:
        job = JOB_CACHE.get(token)
        if job:
            job.update(updates)


def run_analysis_job(token, x_url, lang, redirect_url):
    def progress_callback(stage):
        update_job(token, current_step=STAGE_INDEX.get(stage, 1))

    try:
        analysis = analyze_x_url(x_url, progress_callback=progress_callback)
        message = analysis["message"]
        context_value = analysis["context"]

        result = calculate_analysis(
            input_text=message,
            context=context_value,
            progress_callback=progress_callback,
        )

        payload = {
            "x_url": x_url,
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

    x_url = (request.form.get("x_url") or "").strip()

    errors = {}
    if not x_url:
        errors["x_url"] = text["required_url"]
    elif "twitter.com" not in x_url and "x.com" not in x_url:
        errors["x_url"] = text["url_error"]

    form_values = {
        "x_url": x_url,
    }

    if CLAIMEAI_ERROR:
        errors["backend"] = CLAIMEAI_ERROR

    if errors:
        if is_ajax:
            return jsonify({"ok": False, "errors": errors}), 400
        return render_template("index.html", text=text, lang=lang, form_values=form_values, errors=errors, backend_error=CLAIMEAI_ERROR), 400

    if not is_ajax:
        try:
            analysis = analyze_x_url(x_url)
            if not isinstance(analysis, dict):
                raise RuntimeError(str(analysis))

            result = calculate_analysis(input_text=analysis["message"], context=analysis["context"])

            token = uuid4().hex
            RESULT_CACHE[token] = {
                "x_url": x_url,
                "message": analysis["message"],
                "context": result["context"],
                "verification": result["verification"],
                "patterns": result["patterns"],
                "tone": result["tone"],
                "score": f"{result['score']:.4f}",
                "lang": lang,
            }
            return redirect(url_for("result", token=token, lang=lang))
        except Exception as exc:
            errors["backend"] = str(exc)
            return render_template("index.html", text=text, lang=lang, form_values=form_values, errors=errors, backend_error=CLAIMEAI_ERROR), 400

    token = uuid4().hex
    redirect_url = url_for("result", token=token, lang=lang)

    with JOB_LOCK:
        JOB_CACHE[token] = {
            "x_url": x_url,
            "lang": lang,
            "current_step": 0,
            "done": False,
            "error": None,
            "redirect": redirect_url,
        }

    Thread(
        target=run_analysis_job,
        args=(token, x_url, lang, redirect_url),
        daemon=True,
    ).start()

    return jsonify(
        {
            "ok": True,
            "token": token,
            "status_url": url_for("progress", token=token, lang=lang),
            "redirect": redirect_url,
            "current_step": 0,
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
        x_url=payload.get("x_url"),
        context=payload["context"],
        verification=payload["verification"],
        patterns=payload["patterns"],
        tone=payload["tone"],
        score=payload["score"],
        backend_error=CLAIMEAI_ERROR,
    )


if __name__ == "__main__":
    app.run(debug=True)