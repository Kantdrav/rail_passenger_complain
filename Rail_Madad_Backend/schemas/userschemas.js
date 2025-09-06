import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
      trim: true,
    },
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
    },
    password: {
      type: String,
      required: true,
      minlength: 6,
    },
    phone: {
      type: String,
      required: true,
      match: [/^[0-9]{10}$/, "Please enter a valid 10-digit mobile number"],
    },
    otp: {
      type: String,
      default: null,
    },
    isVerifiedOtp: {
      type: Boolean,
      default: false,
    },
    role: {
      type: String,
      enum: ["user", "admin"], // Future expansion
      default: "user",
    },
  },
  { timestamps: true }
);

const User = mongoose.models.User||mongoose.model("User", userSchema);

export default User;
