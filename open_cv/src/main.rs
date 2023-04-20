use opencv::core::Point;
use opencv::imgproc::filter_2d;
use opencv::prelude::*;
use opencv::{
    highgui::{imshow, named_window, wait_key},
    imgcodecs::{imread, imwrite},
    Result,
};

fn average_filter(img: &Mat, kernel_size: i32) -> Result<Mat, opencv::Error> {
    let mut filtered = Mat::default();
    let kernel_data: &[&[f32]] = &[
        &[1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        &[1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        &[1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
    ];
    let kernel = Mat::from_slice_2d(kernel_data)?;

    let mut filtered = Mat::default();
    filter_2d(
        &img,
        &mut filtered,
        -1,
        &kernel,
        Point { x: -1, y: -1 },
        0.0, // Delta
        opencv::core::BORDER_DEFAULT,
    )?;
    Ok(filtered)
}
fn gaussian_filter(img: &Mat) -> Result<Mat, opencv::Error> {
    let mut filtered = Mat::default();
    let kernel_data: &[&[f32]] = &[
        &[1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
        &[1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0],
        &[1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
    ];
    let kernel = Mat::from_slice_2d(kernel_data)?;

    let mut filtered = Mat::default();
    filter_2d(
        &img,
        &mut filtered,
        -1,
        &kernel,
        Point { x: -1, y: -1 },
        0.0, // Delta
        opencv::core::BORDER_DEFAULT,
    )?;
    Ok(filtered)
}

fn sobel_x_filter(img: &Mat) -> Result<Mat, opencv::Error> {
    let mut filtered = Mat::default();
    let kernel_data: &[&[f32]] = &[&[-1.0, 0.0, 1.0], &[-2.0, 0.0, 2.0], &[-1.0, 0.0, 1.0]];
    let kernel = Mat::from_slice_2d(kernel_data)?;

    let mut filtered = Mat::default();
    filter_2d(
        &img,
        &mut filtered,
        -1,
        &kernel,
        Point { x: -1, y: -1 },
        0.0, // Delta
        opencv::core::BORDER_DEFAULT,
    )?;
    Ok(filtered)
}
fn sobel_y_filter(img: &Mat) -> Result<Mat, opencv::Error> {
    let mut filtered = Mat::default();
    let kernel_data: &[&[f32]] = &[&[-1.0, -2.0, -1.0], &[0.0, 0.0, 0.0], &[1.0, 2.0, 1.0]];
    let kernel = Mat::from_slice_2d(kernel_data)?;

    let mut filtered = Mat::default();
    filter_2d(
        &img,
        &mut filtered,
        -1,
        &kernel,
        Point { x: -1, y: -1 },
        0.0, // Delta
        opencv::core::BORDER_DEFAULT,
    )?;
    Ok(filtered)
}

fn bilateral_filter(img: &Mat) -> Result<Mat, opencv::Error> {
    let mut filtered = Mat::default();
    let kernel_data: &[&[f32]] = &[&[0.0, -1.0, 0.0], &[-1.0, 5.0, -1.0], &[0.0, -1.0, 0.0]];
    let kernel = Mat::from_slice_2d(kernel_data)?;

    let mut filtered = Mat::default();
    filter_2d(
        &img,
        &mut filtered,
        -1,
        &kernel,
        Point { x: -1, y: -1 },
        0.0, // Delta
        opencv::core::BORDER_DEFAULT,
    )?;
    Ok(filtered)
}

fn main() {
    let mut img = Mat::default();
    img = imread("pic_01.jpg", opencv::imgcodecs::IMREAD_COLOR).unwrap();
    if img.empty() {
        panic!("Could not read image");
    }

    named_window("original", opencv::highgui::WINDOW_AUTOSIZE).unwrap();
    imshow("original", &img).unwrap();
    named_window("average filter", opencv::highgui::WINDOW_AUTOSIZE).unwrap();
    named_window("sobel(x) filter", opencv::highgui::WINDOW_AUTOSIZE).unwrap();
    named_window("sobel(y) filter", opencv::highgui::WINDOW_AUTOSIZE).unwrap();
    named_window("gaussian filter", opencv::highgui::WINDOW_AUTOSIZE).unwrap();
    named_window("bilateral filter", opencv::highgui::WINDOW_AUTOSIZE).unwrap();

    let average_filter = average_filter(&img, 5).unwrap();
    imshow("average filter", &average_filter).unwrap();
    imwrite(
        "average_filter.jpg",
        &average_filter,
        &opencv::core::Vector::new(),
    )
    .unwrap();

    let gaussian_filter = gaussian_filter(&img).unwrap();
    imshow("gaussian filter", &gaussian_filter).unwrap();
    imwrite(
        "gaussian_filter.jpg",
        &gaussian_filter,
        &opencv::core::Vector::new(),
    )
    .unwrap();

    let sobel_x_filter = sobel_x_filter(&img).unwrap();
    imshow("sobel(x) filter", &sobel_x_filter).unwrap();
    imwrite(
        "sobel_x_filter.jpg",
        &sobel_x_filter,
        &opencv::core::Vector::new(),
    )
    .unwrap();

    let sobel_y_filter = sobel_y_filter(&img).unwrap();
    imshow("sobel(y) filter", &sobel_y_filter).unwrap();
    imwrite(
        "sobel_y_filter.jpg",
        &sobel_y_filter,
        &opencv::core::Vector::new(),
    )
    .unwrap();

    let bilateral_filter = bilateral_filter(&img).unwrap();
    imshow("bilateral filter", &bilateral_filter).unwrap();
    imwrite(
        "bilateral_filter.jpg",
        &bilateral_filter,
        &opencv::core::Vector::new(),
    )
    .unwrap();

    wait_key(0).unwrap();
}
