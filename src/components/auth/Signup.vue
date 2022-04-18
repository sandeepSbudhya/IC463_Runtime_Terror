<template>
  <v-row>
    <v-col cols="6" offset="4" md="5">
      <v-card width="600px" :justify="center">
        <v-card-title class="title">Sign up for an accident-detection account:</v-card-title>
        <v-form class="pa-3" ref="form" v-model="valid" lazy-validation @submit.prevent="signup()">
          <v-text-field v-model="name" :counter="16" :rules="nameRules" label="Username" required></v-text-field>
          <v-text-field v-model="email" :rules="emailRules" label="Email" required></v-text-field>
          <v-text-field v-model="longitude" :rules="[locationRules]" label="Longitude"></v-text-field>
          <v-text-field v-model="latitude" :rules="[locationRules]" label="Latitude"></v-text-field>
          <v-text-field
            type="password"
            v-model="password"
            :rules="passwordRules"
            label="Password"
            required
          ></v-text-field>
          <span v-if="feedback" class="red--text">{{this.feedback}}</span>
          <br />
          <br />
          <v-btn :disabled="!valid" color="primary" class="mr-4" type="submit">Signup</v-btn>
          <v-btn color="error" class="mr-4" @click="reset">Reset Form</v-btn>
        </v-form>
      </v-card>
    </v-col>
  </v-row>
</template>


<script>
import firebase from "firebase";
import db from "@/firebase/init";
export default {
  name: "Signup",
  data: () => ({
    feedback: "",
    slug: null,
    valid: true,
    name: "",
    credential: null,
    nameRules: [
      (v) => !!v || "Name is required",
      (v) => (v && v.length <= 16) || "Name must be less than 16 characters",
    ],
    email: "",
    emailRules: [
      (v) => !!v || "E-mail is required",
      (v) => /.+@.+\..+/.test(v) || "E-mail must be valid",
    ],
    password: "",
    passwordRules: [(v) => !!v || "Password is required"],
    locationRules: (v) => {
      if (!isNaN(v)) return true;
      return "Enter a number";
    },
    longitude: 0,
    latitude: 0,
  }),

  methods: {
    signup() {
      firebase
        .auth()
        .createUserWithEmailAndPassword(this.email, this.password)
        .then((cred) => {
          this.$root.cred = cred;
          return db
            .collection("users")
            .doc(cred.user.uid)
            .set({
              name: this.name,
              location: {
                longitude: this.longitude,
                latitude: this.latitude,
              },
            });
        })
        .then(() => {
          console.log("successfully written");
          this.$root.authenticated = true;
          this.$root.username = this.name;
          this.$router.push({
            name: "Dash",
          });
        })
        .catch((err) => {
          this.feedback = err;
          console.log(err);
          return;
        });

      //this.$router.push({name: 'Chat',params: {name: this.name}})
    },
    reset() {
      this.$refs.form.reset();
    },
  },
};
</script>