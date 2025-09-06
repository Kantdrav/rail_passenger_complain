import nodemailer from "nodemailer";
import jwt from "jsonwebtoken";
import express from "express";
import bcrypt from "bcryptjs";
import User from "../schemas/userschemas.js";




// 🔑 Generate JWT Token
const generate_token = (user) => {
  return jwt.sign({ id: user._id, role: user.role }, process.env.JWT_TOKEN_SECRET, {
    expiresIn: "5d",
  });
};

// 📩 Function to send OTP email
const sendOtpMail = async (email, otp) => {
  const smtpUser = process.env.SMTP_USER;
  const smtpPass = process.env.SMTP_PASSWORD;

  const transporter = nodemailer.createTransport({
    service: "gmail",
    auth: {
      user: smtpUser,
      pass: smtpPass,
    },
  });

  const mailOptions = {
    from: smtpUser,
    to: email,
    subject: "Rail Madad OTP Verification",
    html: `
      <h2>Welcome to Rail Madad</h2>
      <p>Please verify your account using the OTP below:</p>
      <h3 style="color:#4CAF50">${otp}</h3>
      <p>This OTP is valid for 10 minutes.</p>
    `,
  };

  await transporter.sendMail(mailOptions);
};

// 🟢 REGISTER
export const register = async (req, res) => {
  try {
    console.log("Register endpoint hit");

    const { name, email, password, phone } = req.body;

    // Check existing user
    let existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: "User already exists" });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password.trim(), 10);

    // Generate OTP
    const otp = Math.floor(1000 + Math.random() * 9000);

    // Create new user
    const user = new User({
      name,
      email,
      password: hashedPassword,
      phone,
      otp,
      isVerifiedOtp: false,
    });

    await user.save();

    // Send OTP mail
    await sendOtpMail(email, otp);

    return res.status(201).json({ message: "User registered successfully. Please verify OTP.", user });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: "Error registering user", error });
  }
};

// 🟡 VERIFY OTP
export const verifyOtp = async (req, res) => {
  try {
    const { email, otp } = req.body;

    let user = await User.findOne({ email });
    if (!user) return res.status(400).json({ message: "User not found" });

    if (user.isVerifiedOtp) {
      return res.status(400).json({ message: "User already verified" });
    }

    if (user.otp !== otp) {
      return res.status(400).json({ message: "Invalid OTP" });
    }

    user.isVerifiedOtp = true;
    user.otp = null; // OTP clear after success
    await user.save();

    const token = generate_token(user);

    return res.status(200).json({ message: "User verified successfully", token, user });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: "Error verifying OTP", error });
  }
};

// 🔵 LOGIN
export const login = async (req, res) => {
  try {
    const { email, password } = req.body;

    let user = await User.findOne({ email });
    if (!user) return res.status(401).json({ message: "Invalid email or password" });

    const isValidPassword = await bcrypt.compare(password, user.password);
    if (!isValidPassword) return res.status(401).json({ message: "Invalid email or password" });

    if (!user.isVerifiedOtp) {
      const otp = Math.floor(1000 + Math.random() * 9000);
      user.otp = otp;
      await user.save();

      await sendOtpMail(email, otp);

      return res.status(400).json({ message: "User not verified. OTP sent again." });
    }

    const token = generate_token(user);

    return res.status(200).json({ message: "Login successful", token, user });
  } catch (error) {
    console.error("Login error:", error);
    return res.status(500).json({ message: "Server error", error });
  }
};

