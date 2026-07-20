using System.Text;
using System.Text.Json;

namespace MedDemo.Services;

// Заменяет собой TriageService (который раньше вызывал Claude API) —
// теперь вызываем собственный Python ML-сервис.
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

    public async Task<MlTriageResult> PredictAsync(string symptomsText)
    {
        var request = new { symptoms_text = symptomsText };
        var content = new StringContent(
            JsonSerializer.Serialize(request), Encoding.UTF8, "application/json");

        var response = await _http.PostAsync($"{_mlServiceUrl}/predict", content);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        var result = JsonSerializer.Deserialize<MlTriageResult>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        });

        return result ?? throw new InvalidOperationException("Пустой ответ от ML-сервиса");
    }
}

public class MlTriageResult
{
    public string PredictedDisease { get; set; } = "";
    public string Specialist { get; set; } = "";
    public string Urgency { get; set; } = "";
    public double ConfidenceScore { get; set; }
    public bool IsLowConfidence { get; set; }
    public string ModelType { get; set; } = "";
}