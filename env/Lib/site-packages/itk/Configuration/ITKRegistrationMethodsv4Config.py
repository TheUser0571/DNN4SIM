depends = ('ITKPyBase', 'ITKOptimizersv4', 'ITKMetricsv4', )
templates = (
  ('ImageRegistrationMethodv4Enums', 'itk::ImageRegistrationMethodv4Enums', 'itkImageRegistrationMethodv4Enums', False),
  ('ImageRegistrationMethodv4', 'itk::ImageRegistrationMethodv4', 'itkImageRegistrationMethodv4REGv4F2F2', True, 'itk::Image< float, 2 >, itk::Image< float, 2 >'),
  ('ImageRegistrationMethodv4', 'itk::ImageRegistrationMethodv4', 'itkImageRegistrationMethodv4REGv4D2D2', True, 'itk::Image< double, 2 >, itk::Image< double, 2 >'),
  ('ImageRegistrationMethodv4', 'itk::ImageRegistrationMethodv4', 'itkImageRegistrationMethodv4REGv4F3F3', True, 'itk::Image< float, 3 >, itk::Image< float, 3 >'),
  ('ImageRegistrationMethodv4', 'itk::ImageRegistrationMethodv4', 'itkImageRegistrationMethodv4REGv4D3D3', True, 'itk::Image< double, 3 >, itk::Image< double, 3 >'),
)
snake_case_functions = ('image_registration_methodv4', )
