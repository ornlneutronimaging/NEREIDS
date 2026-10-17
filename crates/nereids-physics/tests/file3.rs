use nereids_endf::parser::parse_endf_file2;
use nereids_endf::resonance::test_support::synthetic_isotope;
use nereids_endf::resonance::{SmoothCrossSections, Tab1};
use nereids_physics::continuous_doppler::{Channel, QuadratureBudget, broaden_with_budget};
use nereids_physics::reich_moore::{CrossSectionPlan, cross_sections_at_energy};

fn tape() -> String {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/endf/Ta-181.endf");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"))
}

fn number(field: &str) -> f64 {
    let field = field.trim();
    if field.is_empty() {
        return 0.0;
    }
    let exponent = field
        .char_indices()
        .skip(1)
        .find(|&(_, c)| c == '+' || c == '-')
        .map(|(i, _)| i);
    match exponent {
        Some(i) if !field[..i].ends_with(['e', 'E']) => format!("{}e{}", &field[..i], &field[i..])
            .parse()
            .expect(field),
        _ => field.parse().expect(field),
    }
}

const MAT: &str = "7328";

/// File 3's section `mt` of material [`MAT`], read field by field.
fn read(text: &str, mt: u32) -> Tab1 {
    let lines: Vec<&str> = text
        .lines()
        .filter(|l| {
            l.len() >= 75
                && l[66..70].trim() == MAT
                && l[70..72].trim() == "3"
                && l[72..75].trim() == mt.to_string()
        })
        .collect();
    let fields =
        |l: &str| -> Vec<f64> { (0..6).map(|i| number(&l[11 * i..11 * i + 11])).collect() };
    let control = fields(lines[1]);
    let (regions, count) = (control[4] as usize, control[5] as usize);
    let rows = (2 * regions).div_ceil(6);
    let laws: Vec<f64> = lines[2..2 + rows].iter().flat_map(|l| fields(l)).collect();
    let values: Vec<f64> = lines[2 + rows..].iter().flat_map(|l| fields(l)).collect();
    Tab1 {
        boundaries: (0..regions).map(|i| laws[2 * i] as usize).collect(),
        interp_codes: (0..regions).map(|i| laws[2 * i + 1] as u32).collect(),
        points: (0..count)
            .map(|i| (values[2 * i], values[2 * i + 1]))
            .collect(),
    }
}

fn linear(points: &[(f64, f64)], energy: f64) -> f64 {
    let i = points.partition_point(|&(e, _)| e <= energy);
    let ((e0, y0), (e1, y1)) = (points[i - 1], points[i]);
    y0 + (energy - e0) / (e1 - e0) * (y1 - y0)
}

#[test]
fn file_3_is_read_as_the_tape_holds_it_and_added_to_each_channel() {
    let text = tape();
    let data = parse_endf_file2(&text).expect("tape");
    let SmoothCrossSections {
        total,
        elastic,
        capture,
        fission,
    } = &data.smooth;
    assert!(fission.is_none());
    for (table, mt) in [(total, 1), (elastic, 2), (capture, 102)] {
        let (table, read) = (table.as_ref().expect("given"), read(&text, mt));
        assert_eq!(
            (&table.boundaries, &table.interp_codes, &table.points),
            (&read.boundaries, &read.interp_codes, &read.points),
            "MT{mt}"
        );
        assert_eq!(read.interp_codes, [2], "MT{mt}");
    }
    let points = read(&text, 2).points;
    let without = nereids_endf::resonance::ResonanceData {
        smooth: SmoothCrossSections::default(),
        ..data.clone()
    };
    let (with, without) = (
        CrossSectionPlan::new(&data),
        CrossSectionPlan::new(&without),
    );
    let mut moved = 0;
    for energy in [2e-4, 0.01, 0.3, 0.9, 20.0] {
        let point = cross_sections_at_energy(&data, energy);
        let (with, without) = (with.evaluate_one(energy), without.evaluate_one(energy));
        assert_eq!(point.elastic, with.elastic, "{energy}");
        let added = with.elastic - without.elastic;
        assert!(
            (added - linear(&points, energy)).abs() <= 1e-12,
            "{energy}: {added}"
        );
        assert_eq!(with.capture, without.capture, "{energy}");
        moved += usize::from(added != 0.0);
    }
    assert!(moved >= 3, "{moved}");
}

/// The tape with `edit` applied to the lines of File 3's section `mt`.
fn edited(mt: &str, edit: impl Fn(&mut Vec<String>)) -> String {
    let mut lines: Vec<String> = tape().lines().map(str::to_string).collect();
    let in_section =
        |l: &String| l.len() >= 75 && l[70..72].trim() == "3" && l[72..75].trim() == mt;
    let first = lines.iter().position(in_section).expect("section");
    let last = lines.iter().rposition(in_section).expect("section");
    let mut section: Vec<String> = lines.drain(first..=last).collect();
    edit(&mut section);
    lines.splice(first..first, section);
    lines.join("\n")
}

#[test]
fn file_3_of_another_material_or_malformed_is_never_read() {
    let other = edited("1", |section| {
        let mut copy: Vec<String> = section
            .iter()
            .map(|l| format!("{}9999{}", &l[..66], &l[70..]))
            .collect();
        copy[3] = format!("{}{:>11}{}", &copy[3][..11], "1.0+3", &copy[3][22..]);
        section.splice(0..0, copy);
    });
    let data = parse_endf_file2(&other).expect("tape");
    assert_eq!(
        data.smooth.total.expect("MT1").points,
        read(&tape(), 1).points
    );
    let extra = edited("1", |section| {
        let last = section[section.len() - 1].clone();
        section.push(last);
    });
    let not_finite = edited("2", |section| {
        section[3] = format!("{}{:>11}{}", &section[3][..11], "nan", &section[3][22..]);
    });
    for (text, expected) in [(extra, "after its table"), (not_finite, "not finite")] {
        let Err(error) = parse_endf_file2(&text) else {
            panic!("{expected}: read");
        };
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn an_evaluation_whose_file_3_holds_the_whole_cross_section_is_refused() {
    let mut lines: Vec<String> = tape().lines().map(str::to_string).collect();
    let head = lines
        .iter()
        .position(|l| l.len() >= 75 && l[70..75] == *" 1451")
        .expect("File 1");
    lines[head] = format!("{}{:>11}{}", &lines[head][..22], 2, &lines[head][33..]);
    let Err(error) = parse_endf_file2(&lines.join("\n")) else {
        panic!("an LRP=2 evaluation was read");
    };
    assert!(error.to_string().contains("LRP=2"), "{error}");
}

#[test]
fn a_one_over_v_file_3_cross_section_broadens_to_itself() {
    let resonances = synthetic_isotope(72, 180, 20.0, 0.01, 0.06);
    let c = 10.0;
    let knots: Vec<(f64, f64)> = [1e-5, 1e-3, 0.1, 1.0, 10.0, 1e3]
        .iter()
        .map(|&e: &f64| (e, c / e.sqrt()))
        .collect();
    let smooth = SmoothCrossSections {
        total: Some(Tab1 {
            boundaries: vec![knots.len()],
            interp_codes: vec![5],
            points: knots,
        }),
        ..SmoothCrossSections::default()
    };
    let with = nereids_endf::resonance::ResonanceData {
        smooth,
        ..resonances.clone()
    };
    let energies = [0.5, 2.0, 5.0, 19.9, 20.0, 80.0];
    let broadened = |data| {
        broaden_with_budget(
            &energies,
            data,
            300.0,
            Channel::Total,
            QuadratureBudget::default(),
        )
        .expect("broadened")
    };
    let (plain, background) = (broadened(&resonances), broadened(&with));
    for (k, &energy) in energies.iter().enumerate() {
        let added = background.values[k] - plain.values[k];
        let expected = c / energy.sqrt();
        let scale = background.values[k];
        assert!(
            (added - expected).abs() <= 1e-7 * scale,
            "{energy}: {added} against {expected}"
        );
        let slope = background.derivatives[k] - plain.derivatives[k];
        assert!(slope.abs() <= 1e-7 * scale / 300.0, "{energy}: {slope}");
    }
}
