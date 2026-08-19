import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from esports.predictor import predict_match, get_all_teams, models_ready, get_analysis_metrics


def home(request):
    return render(request, 'esports/home.html', {
        'teams':        get_all_teams(),
        'models_ready': models_ready(),
    })


@require_POST
def predict_api(request):
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid request body.'}, status=400)

    team_a = data.get('team_a', '').strip()
    team_b = data.get('team_b', '').strip()

    if not team_a or not team_b:
        return JsonResponse({'error': 'Please select both teams.'}, status=400)
    if team_a == team_b:
        return JsonResponse({'error': 'Please select two different teams.'}, status=400)

    try:
        result = predict_match(team_a, team_b)
        return JsonResponse(result)
    except (ValueError, RuntimeError) as exc:
        return JsonResponse({'error': str(exc)}, status=400)


def analysis(request):
    # dict() copy: get_analysis_metrics() is cached and returns a shared object.
    ctx = dict(get_analysis_metrics()) if models_ready() else {}
    ctx['models_ready'] = models_ready()
    return render(request, 'esports/analysis.html', ctx)


def teams_view(request):
    return render(request, 'esports/teams.html', {
        'teams': get_all_teams(),
    })
