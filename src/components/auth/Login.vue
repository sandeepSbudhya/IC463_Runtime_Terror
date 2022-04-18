<template>
  <v-row :align="alignment" :justify="center">
    <v-col cols="6" offset="4" md="5">
      <v-card width="500px" :justify="center">
        <v-form
          class="pa-3"
          ref="form"
          v-model="valid"
          lazy-validation
          v-on:submit.prevent="validate"
        >
          <v-text-field v-model="email" :rules="emailRules" label="Email" required></v-text-field>

          <v-text-field
            type="password"
            v-model="password"
            :rules="passwordRules"
            label="Password"
            required
          ></v-text-field>

          <span class="red--text">{{this.feedback}}</span>
          <br />
          <br />

          <v-btn :disabled="!valid" color="primary" class="mr-4" type="submit">Login</v-btn>

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
  name: "Login",
  data: () => ({
    feedback: "",
    valid: true,
    email: "",
    credential: null,
    emailRules: [
      (v) => !!v || "Email is required",
      (v) => /.+@.+\..+/.test(v) || "E-mail must be valid",
    ],
    password: "",
    passwordRules: [(v) => !!v || "Password is required"],
  }),

  methods: {
    validate() {
      if (this.$refs.form.validate()) {
        if (this.email && this.password) {
          firebase
            .auth()
            .signInWithEmailAndPassword(this.email, this.password)
            .then((cred) => {
              this.$root.cred = cred;
              //find the user record and update that name thing
              return db.collection("users").doc(cred.user.uid).get();
            })
            .then((doc) => {
              this.$root.authenticated = true;
              this.$root.username = doc.data().name;
              this.$router.push({
                name: "Dash",
                query: { credential: this.credential },
              });
            })
            .catch((err) => (this.feedback = err.message));

          this.feedback = null;
        } else {
          this.feedback = "Please enter the fields";
        }
      }
    },
    reset() {
      this.$refs.form.reset();
    },
  },
};
</script>