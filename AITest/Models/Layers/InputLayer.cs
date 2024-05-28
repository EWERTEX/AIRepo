namespace AITest.Models.Layers;

public class InputLayer
{
    public (decimal[], decimal[])[] TrainingSet { get; } =
    [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [1])
    ];
}