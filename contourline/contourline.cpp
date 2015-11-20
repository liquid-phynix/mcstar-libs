#include <contourline/contourline.hpp>

#include <vtk/vtkVersion.h>
#include <vtk/vtkSmartPointer.h>
#include <vtk/vtkPointData.h>
#include <vtk/vtkFloatArray.h>
#include <vtk/vtkCell.h>
#include <vtk/vtkImageData.h>
#include <vtk/vtkMarchingSquares.h>
#include <vtk/vtkStripper.h>

Lines contourlines(float* array, int n0, int n1, float h0, float h1, float value){
    vtkSmartPointer<vtkImageData> id = vtkSmartPointer<vtkImageData>::New();
    id->SetDimensions(n0, n1, 1);
    id->SetSpacing(h0, h1, 1);
    id->SetOrigin(0, 0, 0);
    vtkSmartPointer<vtkFloatArray> fa = vtkSmartPointer<vtkFloatArray>::New();
    fa->SetArray(array, n0 * n1, 1);
    id->GetPointData()->SetScalars(fa);
    vtkSmartPointer<vtkMarchingSquares> cont = vtkSmartPointer<vtkMarchingSquares>::New();
    cont->SetInputData(id);
    cont->SetValue(0, value);
    vtkSmartPointer<vtkStripper> cs = vtkSmartPointer<vtkStripper>::New();
    cs->SetInputConnection(cont->GetOutputPort());
    cs->Update();
    vtkPolyData* pd = cs->GetOutput();

    Lines ret;
    int cells = pd->GetNumberOfCells();
    for(int i = 0; i < cells; i++){
        Line line;
        vtkCell* cell = pd->GetCell(i);
        vtkPoints* points = cell->GetPoints();
        int pointslen = points->GetNumberOfPoints();
        for(int j = 0; j < pointslen; j++){
            double* p = points->GetPoint(j);
            line.emplace_back(Point(p[0], p[1]));
        }
        ret.emplace_back(line);
    }
    return ret;
}
