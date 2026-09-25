using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MedDemo.Services;

// Вызывает Python ML-сервис триажа (FastAPI, POST /predict).
public class MlTriageClient
{
    private readonly HttpClient _http;
    private readonly string _mlServiceUrl;

    public MlTriageClient(HttpClient http, IConfiguration config)
    {
        _http = http;
        // адрес вашего FastAPI сервиса, например http://localhost:8001
        _mlServiceUrl = config["MlService:BaseUrl"]
            ?? throw new InvalidOperationException("MlService:BaseUrl не задан в конфигурации");
    }

    /// <param name="symptomsText">Жалоба пациента.</param>
    /// <param name="language">Язык жалобы: "ru", "uz" или "en". Без него сервис считает текст русским.</param>
    public async Task<MlTriageResult> PredictAsync(string symptomsText, string language = "ru")
    {
        var request = new { symptoms_text = symptomsText, language };
        var content = new StringContent(
            JsonSerializer.Serialize(request), Encoding.UTF8, "application/json");

        var response = await _http.PostAsync($"{_mlServiceUrl}/predict", content);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        var result = JsonSerializer.Deserialize<MlTriageResult>(json);

        return result ?? throw new InvalidOperationException("Пустой ответ от ML-сервиса");
    }
}

// Python отдаёт поля в snake_case (predicted_disease), поэтому имена задаются явно:
// PropertyNameCaseInsensitive не превращает predicted_disease в PredictedDisease.
public class MlTriageResult
{
    [JsonPropertyName("predicted_disease")] public string PredictedDisease { get; set; } = "";
    [JsonPropertyName("specialist")] public string Specialist { get; set; } = "";
    // Итоговая срочность: уже поднята красными флагами, если они сработали.
    [JsonPropertyName("urgency")] public string Urgency { get; set; } = "";
    [JsonPropertyName("model_urgency")] public string ModelUrgency { get; set; } = "";
    [JsonPropertyName("confidence_score")] public double ConfidenceScore { get; set; }
    [JsonPropertyName("is_low_confidence")] public bool IsLowConfidence { get; set; }
    [JsonPropertyName("similarity_score")] public double SimilarityScore { get; set; }
    [JsonPropertyName("is_out_of_distribution")] public bool IsOutOfDistribution { get; set; }
    // Опасные симптомы (например "chest_pain"). Если список не пустой — срочно к врачу.
    [JsonPropertyName("red_flags")] public List<string> RedFlags { get; set; } = new();
    // Главный флаг: если true — не показывать пациенту диагноз, направить к врачу.
    [JsonPropertyName("needs_doctor_review")] public bool NeedsDoctorReview { get; set; }
    // 3 наиболее вероятных диагноза — показывать врачу, а не пациенту.
    [JsonPropertyName("top_candidates")] public List<MlCandidate> TopCandidates { get; set; } = new();
    [JsonPropertyName("translated_text")] public string TranslatedText { get; set; } = "";
    [JsonPropertyName("model_type")] public string ModelType { get; set; } = "";
}

public class MlCandidate
{
    [JsonPropertyName("disease")] public string Disease { get; set; } = "";
    [JsonPropertyName("specialist")] public string Specialist { get; set; } = "";
    [JsonPropertyName("probability")] public double Probability { get; set; }
}
