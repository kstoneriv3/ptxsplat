#include <torch/extension.h>

#include "Ops.h"
#include "Cameras.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    py::enum_<ptxsplat::CameraModelType>(m, "CameraModelType")
        .value("PINHOLE", ptxsplat::CameraModelType::PINHOLE)
        .value("ORTHO", ptxsplat::CameraModelType::ORTHO)
        .value("FISHEYE", ptxsplat::CameraModelType::FISHEYE)
        .value("FTHETA", ptxsplat::CameraModelType::FTHETA)
        .export_values();

    m.def("null", &ptxsplat::null);

    m.def(
        "quat_scale_to_covar_preci_fwd", &ptxsplat::quat_scale_to_covar_preci_fwd
    );
    m.def(
        "quat_scale_to_covar_preci_bwd", &ptxsplat::quat_scale_to_covar_preci_bwd
    );

    m.def("spherical_harmonics_fwd", &ptxsplat::spherical_harmonics_fwd);
    m.def("spherical_harmonics_bwd", &ptxsplat::spherical_harmonics_bwd);

    m.def("adam", &ptxsplat::adam);
    m.def("relocation", &ptxsplat::relocation);

    m.def("intersect_tile", &ptxsplat::intersect_tile);
    m.def("intersect_offset", &ptxsplat::intersect_offset);

    m.def("projection_ewa_simple_fwd", &ptxsplat::projection_ewa_simple_fwd);
    m.def("projection_ewa_simple_bwd", &ptxsplat::projection_ewa_simple_bwd);
    m.def(
        "projection_ewa_3dgs_fused_fwd", &ptxsplat::projection_ewa_3dgs_fused_fwd
    );
    m.def(
        "projection_ewa_3dgs_fused_bwd", &ptxsplat::projection_ewa_3dgs_fused_bwd
    );
    m.def(
        "projection_ewa_3dgs_packed_fwd",
        &ptxsplat::projection_ewa_3dgs_packed_fwd
    );
    m.def(
        "projection_ewa_3dgs_packed_bwd",
        &ptxsplat::projection_ewa_3dgs_packed_bwd
    );

    m.def(
        "rasterize_to_pixels_3dgs_fwd", &ptxsplat::rasterize_to_pixels_3dgs_fwd
    );
    m.def(
        "rasterize_to_pixels_3dgs_bwd", &ptxsplat::rasterize_to_pixels_3dgs_bwd
    );
    m.def("rasterize_to_indices_3dgs", &ptxsplat::rasterize_to_indices_3dgs);

    m.def("projection_2dgs_fused_fwd", &ptxsplat::projection_2dgs_fused_fwd);
    m.def("projection_2dgs_fused_bwd", &ptxsplat::projection_2dgs_fused_bwd);
    m.def("projection_2dgs_packed_fwd", &ptxsplat::projection_2dgs_packed_fwd);
    m.def("projection_2dgs_packed_bwd", &ptxsplat::projection_2dgs_packed_bwd);

    m.def(
        "rasterize_to_pixels_2dgs_fwd", &ptxsplat::rasterize_to_pixels_2dgs_fwd
    );
    m.def(
        "rasterize_to_pixels_2dgs_bwd", &ptxsplat::rasterize_to_pixels_2dgs_bwd
    );
    m.def("rasterize_to_indices_2dgs", &ptxsplat::rasterize_to_indices_2dgs);

    m.def("projection_ut_3dgs_fused", &ptxsplat::projection_ut_3dgs_fused);
    m.def("rasterize_to_pixels_from_world_3dgs_fwd", &ptxsplat::rasterize_to_pixels_from_world_3dgs_fwd);
    m.def("rasterize_to_pixels_from_world_3dgs_bwd", &ptxsplat::rasterize_to_pixels_from_world_3dgs_bwd);

    // Cameras from 3DGUT
    py::enum_<ShutterType>(m, "ShutterType")
        .value("ROLLING_TOP_TO_BOTTOM", ShutterType::ROLLING_TOP_TO_BOTTOM)
        .value("ROLLING_LEFT_TO_RIGHT", ShutterType::ROLLING_LEFT_TO_RIGHT)
        .value("ROLLING_BOTTOM_TO_TOP", ShutterType::ROLLING_BOTTOM_TO_TOP)
        .value("ROLLING_RIGHT_TO_LEFT", ShutterType::ROLLING_RIGHT_TO_LEFT)
        .value("GLOBAL", ShutterType::GLOBAL)
        .export_values();

    py::class_<UnscentedTransformParameters>(m, "UnscentedTransformParameters")
        .def(py::init<>())
        .def_readwrite("alpha", &UnscentedTransformParameters::alpha)
        .def_readwrite("beta", &UnscentedTransformParameters::beta)
        .def_readwrite("kappa", &UnscentedTransformParameters::kappa)
        .def_readwrite("in_image_margin_factor", &UnscentedTransformParameters::in_image_margin_factor)
        .def_readwrite("require_all_sigma_points_valid", &UnscentedTransformParameters::require_all_sigma_points_valid);

    // FTheta Camera support
    py::enum_<FThetaCameraDistortionParameters::PolynomialType>(m, "FThetaPolynomialType")
        .value("PIXELDIST_TO_ANGLE", FThetaCameraDistortionParameters::PolynomialType::PIXELDIST_TO_ANGLE)
        .value("ANGLE_TO_PIXELDIST", FThetaCameraDistortionParameters::PolynomialType::ANGLE_TO_PIXELDIST)
        .export_values();
    py::class_<FThetaCameraDistortionParameters>(m, "FThetaCameraDistortionParameters")
        .def(py::init<>())
        .def_readwrite("reference_poly", &FThetaCameraDistortionParameters::reference_poly)
        .def_readwrite("pixeldist_to_angle_poly", &FThetaCameraDistortionParameters::pixeldist_to_angle_poly)
        .def_readwrite("angle_to_pixeldist_poly", &FThetaCameraDistortionParameters::angle_to_pixeldist_poly)
        .def_readwrite("max_angle", &FThetaCameraDistortionParameters::max_angle)
        .def_readwrite("linear_cde", &FThetaCameraDistortionParameters::linear_cde);
}