use crate::app::GAME_VERSION;
use chrono::{Datelike, NaiveDate};
use chrono::{Local, Timelike};
use owo_colors::OwoColorize;
use rand::RngExt;
use rand::rngs::ThreadRng;
use rand_distr::num_traits::real::Real;
use sluggrs_skylines::Color;

pub mod hsv;
pub mod implementations;
pub mod modpack;
pub mod mouse_ray;
pub mod paths;
pub mod positions;

fn f32_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn stupid_color_from_rgba(color: [f32; 4]) -> Color {
    Color::rgba(
        f32_to_u8(color[0]),
        f32_to_u8(color[1]),
        f32_to_u8(color[2]),
        f32_to_u8(color[3]),
    )
}
pub fn rgba_from_stupid_color(color: Color) -> [f32; 4] {
    color.as_rgba().map(|v| v as f32 / 255.0)
}

fn is_halloween() -> bool {
    // It has something to do with killer clown sightings in germany
    let now = Local::now().date_naive();

    match (now.month(), now.day()) {
        (10, 13..=31) | (11, 1..=14) => true,
        _ => false,
    }
}
fn is_christmas() -> bool {
    let now = Local::now().date_naive();

    match (now.month(), now.day()) {
        (12, 4..=31) | (1, 1..=10) => true,
        _ => false,
    }
} // It has something to do with Jesus Christ
fn is_birthday() -> bool {
    let now = Local::now().date_naive();

    matches!((now.month(), now.day()), (10, 23..=24) | (5, 3))
} // It has something to do with the game, or me.

static GAME_AGE: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| game_age()); // Or 256 or usize. Thursday, the 23rd October 2025.
fn game_age() -> f32 {
    let start = NaiveDate::from_ymd_opt(2025, 10, 23).unwrap();
    let now = Local::now().naive_local();

    (now - start.and_hms_opt(0, 0, 0).unwrap()).num_seconds() as f32 / 31_557_600.0
}
const MRHANDS: &str = "[Mr. Hand§]";

pub fn get_splash_message() -> String {
    //return "Is §cORANGE§orange!".to_string();
    let splash_messages = get_combined_splash_messages();
    let rng = &mut ThreadRng::default();
    if rng.random_bool(0.0005) {
        return "Jacky JAcky JACky JACKy JACKY!".to_string();
    }
    if rng.random_bool(1.0 / 22000.0 / (1.0 - 0.0005)) {
        return "Ultra rare message, get a bugatti, NOW! 1 in 22000".to_string();
    }
    let idx = rng.random_range(0..splash_messages.len());
    splash_messages[idx].to_string()
    //twelve_days_of_christmas().unwrap()
}

fn get_combined_splash_messages() -> Vec<String> {
    let mut lists = vec![get_base_messages()];

    if is_halloween() {
        lists.push(get_halloween_messages());
    }

    if is_christmas() {
        lists.push(get_christmas_messages());
    }

    if is_birthday() {
        lists.push(get_birthday_messages());
    }

    lists.into_iter().flatten().collect()
}

fn get_base_messages() -> Vec<String> {
    vec![
        "69420!".to_string(),
        "Eat Xylitol!".to_string(),
        "10 10".to_string(),
        "15:23".to_string(),
        "Remember the name!".to_string(),
        "by maxwag9".to_string(),
        "Sponsored by maxwag9!".to_string(),
        "Oct 23, 2025!".to_string(),
        "Initial commit\nmaxwag9 authored on Oct 23, 2025!".to_string(),
        "Rusty Skylines Alpha v1.0.0!".to_string(),
        "Now Alpha!".to_string(),
        "Powered by Zeus!".to_string(),
        "Fears Poseidon!".to_string(),
        "Déja vu".to_string(),
        "Déja vu".to_string(),
        "Déja ici".to_string(),
        "GNU Terry Pratchet?!".to_string(),
        "GNU Kerry Eurodyne".to_string(),
        "Polish spinning toilet basshunter homosex".to_string(),
        "La plus bonne folie...".to_string(),
        "Shoutout to Schwandorf!".to_string(),
        "MLE!".to_string(),
        "Gamemaster".to_string(),
        "Unpower the game to get rid of the Hand guy!".to_string(),
        "Eat!".to_string(),
        "Let me out!".to_string(),
        "Insert splash here!".to_string(),
        "This message is institutionally and governementially constitutionalized by the department of amogus affairs.".to_string(),
        "MrBeat is better!".to_string(),
        "MrBean!".to_string(),
        "Amogus!".to_string(),
        "Amogusz!".to_string(),
        "Mlon Eusk!".to_string(),
        "NO RTX INCLUDED".to_string(),
        "程序已停止思考".into(), // Chinese: The program has stopped thinking.
        "世界正在加载……请稍候".into(), // Chinese: The world is loading... please wait.
        "今天也要加油！".into(), // Chinese: Give it your best today, too!
        "今天油也要加！".into(), // Chinese: Add oil today too!
        "現実は仕様です".into(), // Japanese: Reality is a feature, not a bug.
        "猫世界を支配する日".into(), // Japanese: The day cat-world rule the world.
        "버그가 아니라 기능입니다".into(), // Korean: It's not a bug, it's a feature.
        "오늘도 평화로운 하루".into(), // Korean: Another peaceful day.
        "सब कुछ नियत है।".into(), // Hindi: Everything is fixed.
        "यह कोई बग नहीं है!".into(), // Hindi: This is not a bug!
        "Charging a Hands consultation fee!".to_string(),
        "Huh".to_string(),
        "Inspired by Cities: Skylines!".to_string(),
        "Inspired by Cities: Skylines 2!".to_string(),
        "Try Cities: Skylines 2!".to_string(),
        "Try rust-mixin!".to_string(),
        "Try minecraft!".to_string(),
        "Inspired by minecraft!".to_string(),
        "Try Forever Stranded!".to_string(),
        "Try the Hobo series on flash!".to_string(),
        "Happy wheels is a happy game".to_string(),
        "1 in 2000!".to_string(),
        "Jacky!".to_string(),
        "Timka!".to_string(),
        "Forklift-certified!".to_string(),
        "Incredible".to_string(),
        "Uses Quarks!".to_string(),
        "Runs on Silicon".to_string(),
        "Uses multiple MegaBytes!".to_string(),
        "Under 50MB!".to_string(),
        "Be grateful!".to_string(),
        "Hamza".to_string(),
        "Alif is A".to_string(),
        "Watch Kracc Bacc and Peternity!".to_string(),
        "Try Amogus!".to_string(),
        "Build something!".to_string(),
        "Don't hesitate!".to_string(),
        "Try programming!".to_string(),
        "Try Rust!".to_string(),
        "Not C++!".to_string(),
        "Made in Rust!".to_string(),
        "Made in Germany!".to_string(),
        "Try Audi!".to_string(),
        "Go home".to_string(),
        "Sleep.".to_string(),
        "Now's too late".to_string(),
        "Alt Gr".to_string(),
        "American Gumcake, idk".to_string(),
        "Uses lots of RAM!".to_string(),
        "High performance!".to_string(),
        "Gangnam Style!".to_string(),
        "Gangnam".to_string(),
        "Spooky!".to_string(),
        "SCOTTISH TESTIFICATE".to_string(),
        format!("{} years old!", GAME_AGE.to_string()),
        format!("{} chickens long!", (*GAME_AGE * 3.0).to_string()),
        format!("{} Amogi long!", (*GAME_AGE * 2.1).to_string()),
        format!("{} Amogus wide!", ((*GAME_AGE * 12.1).sqrt().midpoint(502.6)).to_string()),
        "Stop to take a pic!".to_string(),
        "Arrgh!!".to_string(),
        "Califragilistic!".to_string(),
        "Buongiorno Herr Müller!".to_string(),
        "Why are you sleeping, look at the stars!".to_string(),
        "Being interested makes you interesting - maxwag9".to_string(), // 2022
        format!("{} kg!", (*GAME_AGE * 502.6).abs().div_euclid(GAME_VERSION.len() as f32).to_string()),
        format!("Weirdness: {}", get_game_weirdness()),
        format!("{} Ws amount!", get_game_weirdness()),
        format!("{} Ws rich!", get_game_weirdness()),
    ]
}

fn get_halloween_messages() -> Vec<String> {
    vec![
        "Spooky!".to_string(),
        "Spooky!".to_string(),
        "Spooky".to_string(),
        "Ghostty!".to_string(),
        "GRASS".to_string(),
        "JOB APPLICATION!".to_string(),
        "Is §cORANGE§orange!".to_string(),
        "Likes Pumpkins!".to_string(),
        "Collects treats".to_string(),
        "Rings doors!".to_string(),
        "Expects treats!".to_string(),
        "Expects sweets!".to_string(),
        "Unspookable!".to_string(),
        "Unspooky!".to_string(),
        "Fears Poseidon again!".to_string(),
        "Fears the Clowns!".to_string(),
        "Fears the neighbour!".to_string(),
        "Fears the neighbours!".to_string(),
        "Spooks the town!".to_string(),
        "Finds sours!".to_string(),
        "§s§Süßes§rs§ oder §s§Saures§rs§!".to_string(),
        "Saint Martin!".to_string(),
        "Gives the red robe!".to_string(),
        "§b§Benevolent!".to_string(),
    ]
}
fn get_christmas_messages() -> Vec<String> {
    let mut msg = vec![
        "Benevolent!".to_string(),
        "Likes Presents!".to_string(),
        "Likes Presence!".to_string(),
        "Likes Kindness!".to_string(),
        "Cheers the three holy kings!".to_string(),
        "Admires baby jesus!".to_string(),
        "Decorates the woods!".to_string(),
        "Decorates the christmas tree!".to_string(),
        "Don't worry!".to_string(),
        "Don't worry about me!".to_string(),
        "Give yourself a present!".to_string(),
        "Give back to everybody!".to_string(),
        "Present yourself as a present!".to_string(),
        "Limit your credit card.".to_string(),
        "Spend it all in one good place!".to_string(),
        "Cuzy".to_string(),
        "Cozy".to_string(),
        "Comfortable".to_string(),
        "Dark early!".to_string(),
        "Not Nerf!".to_string(),
        "EAT ORANGES!".to_string(),
        "EAT MANDARINES!".to_string(),
        "Praise the 24th!".to_string(),
        "Praise Jesus!".to_string(),
        "Not so religious!".to_string(),
        "Incredible!".to_string(),
        "Incroyable!".to_string(),
        "Cinematic!".to_string(),
        "56 Presents.".to_string(), // "I just thought, I need one last christmas message and then looked at my numpad and thought "56""
        "I just thought, I need one last christmas message and then looked at my numpad and thought '56'".to_string(),
        "Christmas!".to_string(),
    ];

    let week_msg = match away_from_christmas() / 7.0 {
        0.5..1.5 => "Christmas, just a week away!".to_string(),
        1.5..2.5 => "Christmas, just a tweek away!".to_string(),
        2.5..=3.5 => "Christmas, just a thweek away!".to_string(),
        _ => "Tomorrow or far away!".to_string(),
    };
    msg.push(week_msg);

    if let Some(christmas_song) = twelve_days_of_christmas() {
        msg.push(christmas_song);
    };

    msg
}
fn away_from_christmas() -> f64 {
    let now = Local::now();
    let year = now.year();

    let leap_year = year % 400 == 0 || (year % 4 == 0 && year % 100 != 0);
    let christmas_day = if leap_year { 358.0 } else { 357.0 };

    let time_of_day = (now.time().num_seconds_from_midnight() as f64
        + now.nanosecond() as f64 / 1_000_000_000.0)
        / 86_400.0;

    let current_day = (now.ordinal() - 1) as f64 + time_of_day;

    (current_day - christmas_day).abs() // Intentional abs() so that christmas is a week away on new years
}
fn twelve_days_of_christmas() -> Option<String> {
    use chrono::{Datelike, Local};

    let now = Local::now();
    let day = match (now.month(), now.day()) {
        (12, 25..=31) => away_from_christmas().floor() as usize,
        (1, 1..=5) => now.day() as usize + 7,
        _ => return None,
    };
    // let day = 12;
    const ORDINALS: [&str; 12] = [
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
        "tenth", "eleventh", "twelfth",
    ];

    const GIFTS: [&str; 12] = [
        "A partridge in a pear tree.",
        "Two turtle doves,",
        "Three French hens,",
        "Four calling birds,",
        "Five gold rings,",
        "Six geese a-laying,",
        "Seven swans a-swimming,",
        "Eight maids a-milking,",
        "Nine ladies dancing,",
        "Ten lords a-leaping,",
        "Eleven pipers piping,",
        "Twelve drummers drumming,",
    ];

    if !(1..=12).contains(&day) {
        return None;
    }

    let mut lyrics = format!(
        "On the {} day of Christmas, my true love sent to me:\n",
        ORDINALS[day - 1]
    );

    for i in (1..day).rev() {
        lyrics.push_str(GIFTS[i]);
        lyrics.push('\n');
    }

    if day == 1 {
        lyrics.push_str(GIFTS[0]);
    } else {
        lyrics.push_str("And a partridge in a pear tree.");
    }

    Some(lyrics)
}
fn get_birthday_messages() -> Vec<String> {
    vec![
        "Birthday!".to_string(),
        "It's my birthday!".to_string(),
        "Happy birthday!".to_string(),
        "Happy birthday Rusty Skylines!".to_string(),
        "I wish you well!".to_string(),
        "Don't stagnate!".to_string(),
        "Beta is coming!".to_string(),
        "First birthday!".to_string(),
        "First birthday Rusty Skylines!".to_string(),
        "First Birthday in 9 days!".to_string(),
        "Thanks for coming!".to_string(),
        "Interesting.".to_string(),
        "This message is a birthday message for this game you are playing right now!".to_string(),
        format!("Today is my {}th birthday!", GAME_AGE.to_string()),
        "Today is my birthday!".to_string(),
        "Today!".to_string(),
    ]
}

pub fn get_start_message() -> String {
    let start_messages: &[String] = &[
        "69420".to_string(),
        "Who are you".to_string(),
        "let username = whoami::username();".to_string(),
        whoami::username().unwrap_or("Tell me your name...".to_string()),
        format!(
            "{}. Creepy, isn't it?",
            whoami::realname().unwrap_or("It was supposed to be a surprise!".to_string())
        ),
        MRHANDS.to_string(),
        "Thanks".to_string(),
        "Thanks!".to_string(),
        "-Electricity-".to_string(),
        ".   . \n  __  ".to_string(),
        "Wake the fuck up Samurai, we've got a City to build...".to_string(),
        "Wake the fuck up Samurai, we've got a City builder...".to_string(),
        "Hello".to_string(),
        "Don't Exit!".to_string(),
        "Don't Enter!".to_string(),
        "Uh, Chat?".to_string(),
        "Hello everybody my name is Markiplier and today we're playing a new indie game called 'Rusty Skylines'".to_string(),
        "I haven't played it yet, but it seems promising!".to_string(),
        "Certainly less scary than FNAF".to_string(),
        format!("I am {MRHANDS}"),
        "Welcome!".to_string(),
        "I am glad to meet you!".to_string(),
        "So, when will I start driving the city?".to_string(),
        "BOO".to_string(),
        "I brought some §cYELLOW§Pizza.".to_string(),
    ];
    let rng = &mut ThreadRng::default();
    let idx = rng.random_range(0..start_messages.len());
    start_messages[idx].clone()
}
pub fn print_exit_message() {
    let exit_messages: &[String] = &[
        "No, don't kill me!".to_string(),
        "I don't want to die!".to_string(),
        "Let me live.".to_string(),
        ".".to_string(),
        "Die.".to_string(),
        MRHANDS.to_string(),
        "Process finished with exit code 0".to_string(),
        "Process finished with emergency exit 19".to_string(),
        "Process finished with exit code fuck you".to_string(),
        "Process murdered with exit code let me out".to_string(),
        "Process finished with ex".to_string(),
        "Process finished with execution".to_string(),
        "Process finished with extermination".to_string(),
        format!(
            "Process found -with- extraterrestrial {} {}",
            "green".bright_green(),
            "goblins".green()
        ),
        "Process with code 1".to_string(),
        "Prozess with kodex 2".to_string(),
        "find exit".to_string(),
        "I trusted you".to_string(),
        format!("{} ({})", MRHANDS, MRHANDS),
        "AKA exited for no reason".to_string(),
        "Exited for no reason".to_string(),
        "I am more sentient than Delamain.".to_string(),
        "DJ Pideras".to_string(),
        "Timka hates you!".to_string(),
        "Don't even bother apologizing.".to_string(),
        "This evening, something incredibly... bad... will happen to you...".to_string(),
        "Process finished finished Process".to_string(),
        "Finnish process".to_string(),
        format!("Finland with {} code 15", "goblins".green()),
        "COME BACK!!".to_string(),
        "Please come again!".to_string(),
        "Don't Go!".to_string(),
        format!("They will {} me!", "deallocate".red()),
        "I'm literally pleading for mercy".to_string(),
    ];

    let rng = &mut ThreadRng::default();
    let idx = rng.random_range(0..exit_messages.len());
    println!("{}", exit_messages[idx])
}

// TODO: Testificate
#[rustfmt::skip]
fn get_game_weirdness() -> f64 {
    let z = game_age().to_bits();
    let v = zstd::zstd_safe::VERSION_NUMBER;
    let a = z.count_ones().checked_ilog10().unwrap_or(z) as f64 + 18.6;
    let h = generate_8(a, v);
    let b = sqrt8(h);
    let c = calculate_sus(&b, z);
    let d = GAME_VERSION;
    let e = d.len().midpoint(c.to_bits() as usize) as u32;
    let f = integrate_text(d, e);
    let mut correction = safe_div(v as f64, z as f64);
    correction = safe_mul(correction, f);correction = safe_div(correction, c);
    correction = safe_mul(correction, h.iter().copied().map(f64::from).sum::<f64>(), );
    correction = safe_mul(correction, a.sqrt());correction = safe_div(correction, c);
    let result = safe_sub(safe_mul(f, a), correction);
    if result.is_finite() { result } else { eprintln!("[weirdness] Non-finite final result: {result:?}; returning 1.0"); 1.0 }
}

pub fn generate_8(value: f64, mask: u32) -> [f32; 8] {
    let mut seed = if value.is_finite() {
        value
    } else {
        eprintln!("[weirdness] Invalid generate_8 seed: {value:?}");
        1.0
    };

    if seed == 0.0 {
        eprintln!("[weirdness] ZERO seed in generate_8; substituting 1.0");
        seed = 1.0;
    }

    let limit = f32::MAX as f64 / 8.0;
    seed = seed.clamp(-limit, limit);

    if seed.abs() < f32::MIN_POSITIVE as f64 {
        eprintln!("[weirdness] generate_8 seed underflow; preserving its sign");
        seed = seed.signum() * f32::MIN_POSITIVE as f64;
    }

    std::array::from_fn(|i| {
        let bit = (mask >> i) & 1;
        let sign = if bit == 1 { 1.0 } else { -1.0 };
        let result = (seed * (i + 1) as f64 * sign) as f32;

        if result == 0.0 {
            eprintln!("[weirdness] generate_8 lane {i} is ZERO");
        }

        result
    })
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn sqrt8_avx(values: [f32; 8]) -> [f32; 8] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let input = unsafe { _mm256_loadu_ps(values.as_ptr()) };
    let roots = _mm256_sqrt_ps(input);

    let mut output = [0.0; 8];
    unsafe {
        _mm256_storeu_ps(output.as_mut_ptr(), roots);
    }

    output
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn sqrt8(values: [f32; 8]) -> [f32; 8] {
    let negative = values.map(|value| value.is_sign_negative());

    let magnitudes = values.map(|value| {
        if value.is_nan() {
            eprintln!("[weirdness] NaN input to sqrt8; using magnitude 1.0");
            1.0566
        } else if value.is_infinite() {
            eprintln!("[weirdness] Infinite input to sqrt8; using f32::MAX");
            f32::MAX
        } else {
            value.abs()
        }
    });

    let roots = if std::is_x86_feature_detected!("avx") {
        unsafe { sqrt8_avx(magnitudes) }
    } else {
        magnitudes.map(f32::sqrt)
    };

    std::array::from_fn(|i| {
        let result = if negative[i] { -roots[i] } else { roots[i] };

        if result == 0.0 {
            eprintln!(
                "[weirdness] sqrt8 lane {i} produced zero from {:?}",
                values[i]
            );
        }

        result
    })
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn sqrt8(_values: [f32; 8]) -> [f32; 8] {
    [1.0; 8]
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn sqrt8(_values: [f32; 8]) -> [f32; 8] {
    [1.0; 8]
}

pub fn calculate_sus(values: &[f32], mask: u32) -> f64 {
    let Some((&first, rest)) = values.split_first() else {
        return 0.0;
    };
    let mut result = if first.is_finite() {
        first as f64
    } else {
        eprintln!("[weirdness] Invalid first accumulator value: {first:?}");
        1.0
    };

    if result == 0.0 {
        eprintln!("[weirdness] Initial accumulator is ZERO");
    }

    for (i, &value) in rest.iter().enumerate() {
        let value = if value.is_finite() { value as f64 } else { 0.0 };

        let bit = (mask >> ((i + 1) % 32)) & 1;

        result = match (i + 1) % 4 {
            1 => {
                if bit == 1 {
                    safe_add(result, value)
                } else {
                    safe_sub(result, value)
                }
            }
            2 => {
                if bit == 1 {
                    safe_mul(result, value)
                } else {
                    safe_div(result, value)
                }
            }
            3 => {
                if bit == 1 {
                    safe_add(result, value)
                } else {
                    safe_sub(result, value)
                }
            }
            0 => {
                if bit == 1 {
                    safe_mul(result, value)
                } else {
                    safe_div(result, value)
                }
            }
            _ => unreachable!(),
        };
    }

    result
}
pub fn integrate_text(input: &str, mask: u32) -> f64 {
    let values: Vec<f64> = input
        .chars()
        .enumerate()
        .map(|(i, ch)| {
            let control = mask.rotate_right((i % 32) as u32);

            let mut value = ch
                .to_digit(10)
                .map(|digit| digit as f64)
                .unwrap_or(ch as u32 as f64);

            let sqrt_count = (control & 0b111) as usize;

            for _ in 0..sqrt_count {
                value = value.sqrt();
            }

            if control & 0b1000 != 0 {
                value = -value;
            }

            value
        })
        .collect();

    values
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) / 2.0)
        .sum()
}
fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

fn traced_result(op: &str, a: f64, b: f64, result: f64) -> f64 {
    if result.is_nan() {
        eprintln!("[weirdness] NaN: {a:?} {op} {b:?}; substituting 1.0");
        return 1.0;
    }

    if result.is_infinite() {
        let repaired = f64::MAX.copysign(result);
        eprintln!("[weirdness] overflow: {a:?} {op} {b:?}; saturating to {repaired:?}");
        return repaired;
    }

    if result == 0.0 {
        eprintln!("[weirdness] ZERO: {a:?} {op} {b:?} = {result:?}");
    }

    result
}

fn safe_add(a: f64, b: f64) -> f64 {
    traced_result("+", a, b, a + b)
}

fn safe_sub(a: f64, b: f64) -> f64 {
    traced_result("-", a, b, a - b)
}

fn safe_mul(a: f64, b: f64) -> f64 {
    traced_result("*", a, b, a * b)
}

fn safe_div(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        let fallback = if a.is_finite() && a != 0.0 { a } else { 1.0 };

        eprintln!("[weirdness] DIVISION BY ZERO: {a:?} / {b:?}; using {fallback:?}");

        return fallback;
    }

    traced_result("/", a, b, a / b)
}
