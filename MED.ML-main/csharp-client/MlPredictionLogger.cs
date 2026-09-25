using Microsoft.Data.SqlClient;
using System.Text.Json;

namespace MedDemo.Services;

// Пример записи результата ML-предсказания в вашу СУЩЕСТВУЮЩУЮ таблицу MLPredictions.
// Подставьте реальные имена колонок, если они отличаются от того, что было на скриншоте
// (Id, ModelType, EntityType, EntityId, InputSnapshot, PredictedValue, ConfidenceScore,
//  CreatedAt, Feedback).
public class MlPredictionLogger
{
    private readonly string _connectionString;

    public MlPredictionLogger(IConfiguration config)
    {
        _connectionString = config.GetConnectionString("Default")
            ?? throw new InvalidOperationException("ConnectionStrings:Default не задан");
    }

    public async Task<int> LogPredictionAsync(
        string entityType,      // например "consultation"
        int entityId,           // id консультации
        string symptomsText,
        MlTriageResult prediction)
    {
        using var connection = new SqlConnection(_connectionString);
        await connection.OpenAsync();

        var inputSnapshot = JsonSerializer.Serialize(new
        {
            symptoms_text = symptomsText,
            translated_text = prediction.TranslatedText
        });
        var predictedValue = JsonSerializer.Serialize(new
        {
            disease = prediction.PredictedDisease,
            specialist = prediction.Specialist,
            urgency = prediction.Urgency,
            similarity_score = prediction.SimilarityScore,
            is_out_of_distribution = prediction.IsOutOfDistribution,
            needs_doctor_review = prediction.NeedsDoctorReview,
            model_urgency = prediction.ModelUrgency,
            red_flags = prediction.RedFlags,
            top_candidates = prediction.TopCandidates
        });

        var cmd = new SqlCommand(@"
            INSERT INTO MLPredictions
                (ModelType, EntityType, EntityId, InputSnapshot, PredictedValue, ConfidenceScore, CreatedAt)
            OUTPUT INSERTED.Id
            VALUES (@modelType, @entityType, @entityId, @inputSnapshot, @predictedValue, @confidence, SYSUTCDATETIME())",
            connection);

        cmd.Parameters.AddWithValue("@modelType", prediction.ModelType);
        cmd.Parameters.AddWithValue("@entityType", entityType);
        cmd.Parameters.AddWithValue("@entityId", entityId);
        cmd.Parameters.AddWithValue("@inputSnapshot", inputSnapshot);
        cmd.Parameters.AddWithValue("@predictedValue", predictedValue);
        cmd.Parameters.AddWithValue("@confidence", prediction.ConfidenceScore);

        return (int)(await cmd.ExecuteScalarAsync())!;
    }

    // Вызывается позже, когда врач подтвердил или поправил рекомендацию —
    // это и есть данные для будущего дообучения модели.
    public async Task UpdateFeedbackAsync(int predictionId, string feedback)
    {
        using var connection = new SqlConnection(_connectionString);
        await connection.OpenAsync();

        var cmd = new SqlCommand(
            "UPDATE MLPredictions SET Feedback = @feedback WHERE Id = @id", connection);
        cmd.Parameters.AddWithValue("@feedback", feedback);
        cmd.Parameters.AddWithValue("@id", predictionId);
        await cmd.ExecuteNonQueryAsync();
    }
}