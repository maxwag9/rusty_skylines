use chrono::Local;
use chrono::{Datelike, NaiveDate};
use rand::RngExt;
use rand::rngs::ThreadRng;
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
    let now = Local::now().date_naive();
    let month = now.month();
    let day = now.day();

    (month == 10 && day >= 13) || (month == 11 && day <= 10)
}
fn is_christmas() -> bool {
    let now = Local::now().date_naive();
    let month = now.month();
    let day = now.day();

    (month == 12 && day >= 4) || (month == 1 && day <= 10)
}
fn is_birthday() -> bool {
    let now = Local::now().date_naive();
    let month = now.month();
    let day = now.day();

    (month == 10 && day >= 23) || (month == 10 && day <= 24)
}
static GAME_AGE: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| {
    let start = NaiveDate::from_ymd_opt(2025, 10, 23).unwrap();
    let now = Local::now().naive_local();

    (now - start.and_hms_opt(0, 0, 0).unwrap()).num_seconds() as f32 / 31_557_600.0
}); // Or 256 or usize. Thursday, the 23rd October 2025.
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
    let idx = rng.random_range(1..splash_messages.len());
    splash_messages[idx].to_string()
}

fn get_combined_splash_messages() -> Vec<String> {
    let lists = vec![
        get_base_messages(),
        get_halloween_messages(),
        get_christmas_messages(),
        get_birthday_messages(),
    ];
    lists.into_iter().flatten().collect::<Vec<String>>()
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
    vec![
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
        "I just thought, I need one last christmas message and then looked at my numpad and thought '56'".to_string()
    ]
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
    ]
}

pub fn print_start_message() {
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
    ];
    let rng = &mut ThreadRng::default();
    let idx = rng.random_range(0..start_messages.len());
    println!("{}", start_messages[idx])
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
        "Process found -with- extraterrestrial green goblins".to_string(),
        "Process with code 1".to_string(),
        "Prozess with kodex 2".to_string(),
        "find exit".to_string(),
        "I trusted you".to_string(),
        format!("{} ({})", MRHANDS, MRHANDS),
    ];

    let rng = &mut ThreadRng::default();
    let idx = rng.random_range(0..exit_messages.len());
    println!("{}", exit_messages[idx])
}

// TODO: Testificate
